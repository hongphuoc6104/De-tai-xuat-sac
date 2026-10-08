"""Colab-only training entry points. Importing this module never starts training."""
import gc
import json
import os
import random
import re
import tarfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from data_integrity import (metadata_table, validate_frame, split_patients, assert_disjoint,
                            patient_predictions, file_hash, fingerprint, atomic_json, lock_config)

PIPELINE_VERSION = 'precut-pretrained-v2'
IMAGE_EXTENSIONS = {'.png','.jpg','.jpeg','.tif','.tiff'}
METHODS = {'A0_baseline':(True,False), 'A1_clahe_on':(True,True),
           'A2_no_percentile':(False,False), 'A3_clahe_no_percentile':(False,True)}


def image_files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def list_drive_tiles(service, folder_id):
    """Paginate every folder. Missing pages/errors never become a completed inventory."""
    pending=[(folder_id,Path())]; entries=[]; seen=set(); paths=set()
    while pending:
        parent,prefix=pending.pop()
        if parent in seen: raise ValueError('Repeated/cyclic Drive folder.')
        seen.add(parent); token=None
        while True:
            response=service.files().list(q=f"'{parent}' in parents and trashed = false",
                pageSize=1000,pageToken=token,
                fields='nextPageToken,incompleteSearch,files(id,name,mimeType,size,md5Checksum)',
                supportsAllDrives=True,includeItemsFromAllDrives=True).execute(num_retries=5)
            if response.get('incompleteSearch'): raise RuntimeError('Drive returned an incomplete search.')
            for entry in response.get('files',[]):
                name=entry['name']
                if name in ['.','..'] or '/' in name or '\\' in name:
                    raise ValueError('Unsafe filename in Drive folder.')
                rel=prefix/name
                if entry['mimeType']=='application/vnd.google-apps.folder':
                    pending.append((entry['id'],rel))
                elif Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                    if rel.as_posix() in paths: raise ValueError('Duplicate Drive file paths.')
                    paths.add(rel.as_posix())
                    if not entry.get('md5Checksum') or not entry.get('size'):
                        raise ValueError('Drive image has no size/checksum.')
                    entries.append(dict(entry,path=rel.as_posix()))
            token=response.get('nextPageToken')
            if not token: break
    if not entries: raise ValueError('Drive folder contains no accessible tile images.')
    return sorted(entries,key=lambda e:e['path'])


def download_drive_tiles(source_url,root):
    import hashlib
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    match=re.search(r'/folders/([A-Za-z0-9_-]+)',source_url)
    if not match: raise ValueError('Expected a Google Drive folder URL.')
    auth.authenticate_user()
    credentials,_=google.auth.default()
    service=build('drive','v3',credentials=credentials,cache_discovery=False)
    entries=list_drive_tiles(service,match.group(1))
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    def valid(path,entry):
        if not path.is_file() or path.stat().st_size!=int(entry['size']): return False
        h=hashlib.md5()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()==entry['md5Checksum']
    from tqdm.auto import tqdm
    for entry in tqdm(entries,desc='Download existing Drive tiles'):
        target=root/entry['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if valid(target,entry): continue
        temp=target.with_name(target.name+'.part')
        request=service.files().get_media(fileId=entry['id'],supportsAllDrives=True)
        with temp.open('wb') as f:
            downloader=MediaIoBaseDownload(f,request,chunksize=1024*1024)
            done=False
            while not done: _,done=downloader.next_chunk(num_retries=5)
        if not valid(temp,entry): raise ValueError('Downloaded tile checksum mismatch: '+entry['path'])
        os.replace(temp,target)
    # Only the successfully enumerated files enter this snapshot (ignore stale staging files).
    return [root/e['path'] for e in entries]


def prepare_precut_cache(source_url, cache_root, local_root, source_dir=None):
    """One persistent raw-tile archive; never calls a tile cutter or preprocessing script."""
    cache_root, local_root = Path(cache_root), Path(local_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / 'precut_tiles.tar'
    inventory_path = cache_root / 'inventory.json'
    source = {'url': source_url, 'source_dir': str(source_dir) if source_dir else None}
    if not archive.exists() or not inventory_path.exists():
        if source_dir:
            root = Path(source_dir)
            if not root.is_dir():
                raise FileNotFoundError(f'PRE_CUT_SOURCE_DIR does not exist: {root}')
            files = image_files(root)
        else:
            root = cache_root / 'download'
            print('Downloading existing tiles once into persistent Drive cache. No cutting.')
            try:
                files = download_drive_tiles(source_url, root)
            except Exception as exc:
                raise RuntimeError('Drive download did not complete. Cache is not marked ready. '
                    'Retry, or add the tile folder shortcut to MyDrive and set PRE_CUT_SOURCE_DIR.') from exc
        if not files:
            raise ValueError('No images found in the precut source.')
        from PIL import Image
        entries = []
        temp = archive.with_suffix('.tar.tmp')
        with tarfile.open(temp, 'w') as tf:
            for p in files:
                # A failed/HTML/truncated download cannot become a valid cache.
                with Image.open(p) as im:
                    im.verify()
                rel = p.relative_to(root).as_posix()
                entries.append({'path':rel, 'size':p.stat().st_size, 'sha256':file_hash(p)})
                tf.add(p, arcname=rel, recursive=False)
        os.replace(temp, archive)
        atomic_json(inventory_path, {'source':source, 'files':entries})
        if source_dir is None:
            shutil.rmtree(root)  # only the downloader-owned staging copy; archive is committed
    inventory = json.loads(inventory_path.read_text())
    if inventory['source'] != source:
        raise ValueError('Cache belongs to another data source. Choose a different CACHE_ROOT.')
    token = fingerprint(inventory)
    local_root.mkdir(parents=True, exist_ok=True)
    marker = local_root / '.cache_ready.json'
    ready = marker.exists() and json.loads(marker.read_text()).get('fingerprint') == token
    ready = ready and all((local_root / e['path']).is_file() and
                          (local_root / e['path']).stat().st_size == e['size'] for e in inventory['files'])
    if not ready:
        print('Restoring cached tiles from Drive archive to Colab SSD. No cutting.')
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                target = (local_root / member.name).resolve()
                if not target.is_relative_to(local_root.resolve()) or not member.isfile():
                    raise ValueError('Unsafe archive member.')
            tf.extractall(local_root, filter='data')
        for e in inventory['files']:
            if file_hash(local_root / e['path']) != e['sha256']:
                raise ValueError(f"Cache checksum mismatch: {e['path']}")
        atomic_json(marker, {'fingerprint':token})
    print(f"Ready: {len(inventory['files'])} precut images on SSD; persistent archive: {archive}")
    return inventory


def build_manifest(root, metadata_path, patient_columns, inventory, audit_dir, unmatched_policy="error"):
    if unmatched_policy not in {"error", "exclude"}:
        raise ValueError("unmatched_policy must be error or exclude.")
    meta = metadata_table(metadata_path, patient_columns).set_index('slide_id')
    rows, unmatched = [], []
    for entry in inventory['files']:
        p = Path(root) / entry['path']
        if file_hash(p) != entry['sha256']:
            raise ValueError(f'Local tile changed/corrupted: {p}')
        # Actual Drive layout: IMG_<timestamp>/<x>_<y>.png.
        # Parent directory matching is exact; never use an ambiguous string prefix.
        candidates = set(part for part in Path(entry['path']).parts[:-1] if part in meta.index)
        if not candidates and p.stem in meta.index:
            candidates = {p.stem}
        if len(candidates) != 1:
            unmatched.append(entry['path'])
            continue
        sid = candidates.pop()
        row = meta.loc[sid].to_dict()
        rows.append(dict(row, slide_id=sid, image_path=str(p), tile_path=entry['path'], sha256=entry['sha256']))
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    excluded = pd.DataFrame({'unmatched_image':unmatched})
    excluded.to_csv(audit_dir/'unmatched_images.csv',index=False)
    if unmatched:
        excluded['folder'] = excluded.unmatched_image.map(lambda p:Path(p).parent.as_posix())
        excluded.groupby('folder').size().rename('excluded_tiles').to_csv(audit_dir/'unmatched_folders.csv')
        if unmatched_policy == 'error':
            raise ValueError(f'{len(unmatched)} tiles do not map uniquely to metadata. See unmatched_images.csv; no guessed labels.')
        print(f'EXCLUDED {len(unmatched)} tiles without unique metadata; see {audit_dir}/unmatched_folders.csv')
    atomic_json(audit_dir/'metadata_coverage.json', {'total_tiles':len(inventory['files']),
                'matched_tiles':len(rows),'excluded_tiles':len(unmatched),'unmatched_policy':unmatched_policy})
    df = pd.DataFrame(rows)
    validate_frame(df)
    # Same bytes in another patient would defeat patient separation; fail before splitting.
    if (df.groupby('sha256').patient_id.nunique() > 1).any():
        raise ValueError('Identical image bytes assigned to different patients. Resolve duplicates before training.')
    if (df.groupby('sha256').label.nunique() > 1).any():
        raise ValueError('Identical image bytes have conflicting labels.')
    df = df.drop_duplicates('sha256').sort_values('tile_path').reset_index(drop=True)
    df.to_csv(audit_dir/'dataset_manifest.csv', index=False)
    return df


def fixed_holdout(df, folder, ratio, seed):
    """Persistent global patient split, shared by all architectures and magnifications."""
    folder = Path(folder)
    lock_config(folder, {'dataset':fingerprint(df[['tile_path','patient_id','label','patient_label','sha256']].to_dict('records')),
                         'test_ratio':ratio,'split_seed':seed,'version':PIPELINE_VERSION})
    train, test = split_patients(df, ratio, seed)
    path = folder/'patient_split.csv'
    assignment = pd.concat([train.assign(split='train'),test.assign(split='test')])
    assignment = assignment[['patient_id','split']].drop_duplicates().sort_values('patient_id')
    if path.exists():
        old = pd.read_csv(path,dtype=str).sort_values('patient_id').reset_index(drop=True)
        if not old.equals(assignment.reset_index(drop=True)):
            raise ValueError('Saved holdout differs from requested split.')
    else:
        assignment.to_csv(path,index=False)
    return train, test


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    import torch
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_model(name, pretrained=True, image_size=224):
    import torch
    from torch import nn
    from torchvision.models import (efficientnet_b0, convnext_tiny, vit_b_16,
                                    EfficientNet_B0_Weights, ConvNeXt_Tiny_Weights, ViT_B_16_Weights)
    def backbone(kind):
        if kind == 'efficientnet':
            m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            dim = m.classifier[1].in_features
            m.classifier[1] = nn.Identity()
        elif kind == 'convnext':
            m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            dim = m.classifier[2].in_features
            m.classifier[2] = nn.Identity()
        elif kind == 'vit':
            if image_size != 224:
                raise ValueError('Pretrained ViT uses IMG_SIZE=224 in this pipeline.')
            m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None, image_size=224)
            dim = m.heads.head.in_features
            m.heads.head = nn.Identity()
        else:
            raise ValueError(f'Unknown model: {kind}')
        return m, dim
    class BinaryModel(nn.Module):
        def __init__(self):
            super().__init__()
            kinds = ['efficientnet','convnext','vit'] if name == 'ensemble' else [name]
            built = [backbone(k) for k in kinds]
            self.backbones = nn.ModuleList([m for m,_ in built])
            dim = sum(d for _,d in built)
            self.head = nn.Sequential(nn.Linear(dim,256),nn.LayerNorm(256),nn.ReLU(),
                                      nn.Dropout(.3),nn.Linear(256,1)) if name == 'ensemble' else nn.Linear(dim,1)
        def forward(self,x):
            return self.head(torch.cat([m(x) for m in self.backbones],dim=1)).squeeze(1)
    return BinaryModel()


def make_loader(df, config, train=False, seed=42):
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from PIL import Image
    from torchvision.transforms import functional as TF, InterpolationMode
    percentile, clahe_on = METHODS[config['experiment']]
    class Tiles(Dataset):
        def __len__(self): return len(df)
        def __getitem__(self, idx):
            with Image.open(df.iloc[idx].image_path) as im:
                img = im.convert('RGB')
            # One deterministic preprocessing route for both train and inference.
            img = TF.resize(img, [config['img_size'],config['img_size']], interpolation=InterpolationMode.BILINEAR, antialias=True)
            arr = np.asarray(img).copy()
            if percentile:
                out = arr.astype(np.float32)
                for c in range(3):
                    low,high = np.percentile(out[...,c],[1,99])
                    if high-low > 1e-6:
                        out[...,c] = np.clip((out[...,c]-low)*255/(high-low),0,255)
                arr = out.astype(np.uint8)
            if clahe_on:
                import cv2
                lab = cv2.cvtColor(arr,cv2.COLOR_RGB2LAB)
                lab[...,0] = cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[...,0])
                arr = cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
            img = Image.fromarray(arr)
            if train:
                if random.random() < .5: img = TF.hflip(img)
                if random.random() < .5: img = TF.vflip(img)
                img = TF.rotate(img,90*random.randrange(4))
            x = TF.normalize(TF.to_tensor(img),[.485,.456,.406],[.229,.224,.225])
            return x, torch.tensor(float(df.iloc[idx].label))
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if train and config['balance'] == 'weighted_sampler':
        counts = df.label.value_counts()
        weights = df.label.map(lambda y:1/counts[y]).to_numpy()
        sampler = WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),len(df),replacement=True,generator=generator)
    elif config['balance'] not in ['none','class_weight','weighted_sampler']:
        raise ValueError('Invalid balance mode.')
    return DataLoader(Tiles(),batch_size=config['batch_size'],shuffle=train and sampler is None,
                      sampler=sampler,num_workers=config['num_workers'],pin_memory=True,
                      worker_init_fn=seed_worker,generator=generator)


def metrics(y, p, threshold=.5):
    from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score
    y,p = np.asarray(y,int),np.asarray(p,float)
    if not len(y) or set(np.unique(y)) != {0,1} or not np.isfinite(p).all():
        raise ValueError('Metrics require finite predictions and both classes.')
    pred = p >= threshold
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens,spec = tp/(tp+fn),tn/(tn+fp)
    return {'auc':float(roc_auc_score(y,p)), 'accuracy':float((tp+tn)/len(y)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'sensitivity':float(sens),'specificity':float(spec),
            'balanced_accuracy':float((sens+spec)/2), 'threshold':float(threshold)}


def choose_threshold(y,p):
    # Validation only; freeze before looking at test predictions.
    choices = [(float(t),metrics(y,p,t)['balanced_accuracy']) for t in np.linspace(0,1,101)]
    return sorted(choices,key=lambda x:(-x[1],abs(x[0]-.5)))[0][0]


def predict(model,df,config,device):
    import torch
    loader = make_loader(df,config)
    probs=[]
    model.eval()
    with torch.no_grad():
        for x,_ in loader:
            probs.extend(torch.sigmoid(model(x.to(device))).cpu().tolist())
    out=df.reset_index(drop=True).copy()
    out['y_prob']=probs
    return out


def atomic_torch_save(obj,path):
    import torch
    path=Path(path)
    temp=path.with_name(path.name+'.tmp')
    torch.save(obj,temp)
    os.replace(temp,path)


def capture_rng():
    import torch
    n = np.random.get_state()
    return {'python':random.getstate(), 'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    n=state['numpy']
    np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def train_seed(train,val,config,folder,seed):
    import torch
    assert_disjoint(train,val)
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires Colab GPU. CPU training is disabled.')
    seed_all(seed)
    folder=Path(folder)
    run_config=dict(config,seed=seed,train_ids=sorted(train.tile_path.tolist()),val_ids=sorted(val.tile_path.tolist()))
    lock_config(folder,run_config)
    latest,best = folder/'latest_checkpoint.pt',folder/'best_model.pt'
    checkpoint = torch.load(latest,map_location='cpu',weights_only=True) if latest.exists() else None
    if checkpoint is not None and checkpoint['completed']:
        if not best.exists(): raise FileNotFoundError('Completed run missing best_model.pt')
        return best
    device=torch.device('cuda')
    model=build_model(config['model'],pretrained=checkpoint is None,image_size=config['img_size']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'])
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=(torch.tensor([(train.label==0).sum()/max(1,(train.label==1).sum())],device=device)
                                                   if config['balance']=='class_weight' else None))
    scaler=torch.amp.GradScaler('cuda',enabled=config['amp'])
    history=[]; best_auc=-1.; best_epoch=0; no_improve=0; start=1
    if checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        history=checkpoint['history']; best_auc=checkpoint['best_auc']; best_epoch=checkpoint['best_epoch']
        no_improve=checkpoint['no_improve']; start=checkpoint['epoch']+1
        restore_rng(checkpoint['rng'])
        if best_epoch and not best.exists(): raise FileNotFoundError('Resume requires existing best_model.pt')
        del checkpoint
    from tqdm.auto import tqdm
    for epoch in range(start,config['epochs']+1):
        loader=make_loader(train,config,train=True,seed=seed*10000+epoch)
        model.train(); total=0.; count=0
        for x,y in tqdm(loader,desc=f"{config['model']} seed={seed} epoch={epoch}"):
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=config['amp']):
                loss=criterion(model(x),y)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite train loss; checkpoint not advanced.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer); scaler.update()
            total+=float(loss.detach())*len(y); count+=len(y)
        val_tiles=predict(model,val,config,device)
        pat=patient_predictions(val_tiles)
        m=metrics(pat.y_true,pat.y_prob)
        history.append({'epoch':epoch,'train_loss':total/count,'val_patient_auc':m['auc'],'val_patient_f1':m['f1']})
        improved=m['auc'] > best_auc+config['min_delta']
        if improved:
            best_auc=m['auc'];best_epoch=epoch;no_improve=0
            atomic_torch_save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},best)
        else: no_improve+=1
        scheduler.step()
        completed=no_improve>=config['patience'] or epoch>=config['epochs']
        atomic_torch_save({'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                          'scheduler_state':scheduler.state_dict(),'scaler_state':scaler.state_dict(),
                          'history':history,'best_auc':best_auc,'best_epoch':best_epoch,'no_improve':no_improve,
                          'completed':completed,'rng':capture_rng()},latest)
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        print(f"Epoch {epoch}: loss={total/count:.4f}, val patient AUC={m['auc']:.4f}; best epoch={best_epoch}")
        if completed: break
    if not best.exists(): raise RuntimeError('No valid best checkpoint; refusing test evaluation.')
    del model,optimizer,scaler
    gc.collect();torch.cuda.empty_cache()
    return best


def calibrate_seed(checkpoint,val,config,folder):
    import torch
    model=build_model(config['model'],pretrained=False,image_size=config['img_size']).cuda()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    tiles=predict(model,val,config,'cuda');pat=patient_predictions(tiles)
    thresholds={'tile':choose_threshold(tiles.label,tiles.y_prob),'patient':choose_threshold(pat.y_true,pat.y_prob),
                'checkpoint_sha256':file_hash(checkpoint)}
    atomic_json(Path(folder)/'thresholds_from_validation.json',thresholds)
    tiles.to_csv(Path(folder)/'val_tile_predictions.csv',index=False)
    pat.to_csv(Path(folder)/'val_patient_predictions.csv',index=False)
    del model;gc.collect();torch.cuda.empty_cache()
    return thresholds


def test_seed(checkpoint,test,config,folder):
    import torch
    validate_frame(test,'independent test')
    folder=Path(folder)
    thresholds=json.loads((folder/'thresholds_from_validation.json').read_text())
    if thresholds['checkpoint_sha256'] != file_hash(checkpoint):
        raise ValueError('Checkpoint changed after threshold calibration.')
    model=build_model(config['model'],pretrained=False,image_size=config['img_size']).cuda()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    tiles=predict(model,test,config,'cuda');pat=patient_predictions(tiles)
    tiles.to_csv(folder/'test_tile_predictions.csv',index=False)
    pat.to_csv(folder/'test_patient_predictions.csv',index=False)
    result={**{'tile_'+k:v for k,v in metrics(tiles.label,tiles.y_prob,thresholds['tile']).items()},
            **{'patient_'+k:v for k,v in metrics(pat.y_true,pat.y_prob,thresholds['patient']).items()}}
    atomic_json(folder/'test_metrics.json',result)
    del model;gc.collect();torch.cuda.empty_cache()
    return result


def prepare_experiment(df,config,output_root):
    validate_frame(df)
    if config['experiment'] not in METHODS: raise ValueError('Unknown preprocessing experiment.')
    if config['img_size'] != 224: raise ValueError('Use img_size=224 for consistent pretrained comparisons.')
    full_train,full_test=fixed_holdout(df,Path(output_root)/'shared_holdout',config['test_ratio'],config['split_seed'])
    mag=config['magnification']
    train=full_train[full_train.objective_lens==mag].copy()
    test=full_test[full_test.objective_lens==mag].copy()
    assert_disjoint(train,test)
    config=dict(config,version=PIPELINE_VERSION,weights='IMAGENET1K_V1',
                runtime_sha256=file_hash(__file__), integrity_sha256=file_hash(Path(__file__).with_name('data_integrity.py')),
                dataset=fingerprint(df[['tile_path','patient_id','label','sha256','objective_lens']].to_dict('records')))
    tag=f"{config['model']}_{config['experiment']}_{mag}X_{fingerprint(config)[:12]}"
    folder=Path(output_root)/('smoke' if config['smoke_test'] else 'experiments')/tag
    lock_config(folder,config)
    assignments={}
    for seed in config['seeds']:
        tr,va=split_patients(train,config['val_ratio'],seed)
        assert_disjoint(tr,va,test)
        run=folder/f'seed_{seed}'; run.mkdir(parents=True,exist_ok=True)
        for name,frame in [('train',tr),('val',va),('test',test)]:
            frame.to_csv(run/f'{name}_manifest.csv',index=False)
        assignments[seed]=(tr,va,test,run)
    print('Preflight passed. Patient counts: train pool=',train.patient_id.nunique(), 'test=',test.patient_id.nunique())
    return config,folder,assignments


def train_experiment(config,assignments):
    # This is called only by the explicit training notebook cell on Colab.
    for seed,(tr,va,_,folder) in assignments.items():
        checkpoint=train_seed(tr,va,config,folder,seed)
        calibrate_seed(checkpoint,va,config,folder)
    print('Training and validation calibration complete. Test data has not been evaluated.')


def evaluate_experiment(config,folder,assignments):
    if config['smoke_test']:
        raise ValueError('Smoke runs cannot produce scientific test reports.')
    results=[]
    for seed,(_,_,test,run) in assignments.items():
        results.append(dict(seed=seed,**test_seed(run/'best_model.pt',test,config,run)))
    out=pd.DataFrame(results)
    out.to_csv(Path(folder)/'final_test_metrics_summary.csv',index=False)
    summary=out.drop(columns='seed').agg(['mean','std'])
    summary.to_csv(Path(folder)/'test_mean_std_across_seeds.csv')
    print('Mean/std below describe variation across training seeds on the SAME patient test set, not a confidence interval.')
    print(summary)
    # Fixed threshold declared before test: seed models have different validation patients,
    # so their validation predictions cannot be pooled to tune a seed-ensemble threshold.
    parts=[]
    for seed,(_,_,_,run) in assignments.items():
        frame=pd.read_csv(run/'test_patient_predictions.csv',dtype={'patient_id':str})
        frame['seed']=seed
        parts.append(frame)
    combined=pd.concat(parts)
    if not combined.groupby('patient_id').seed.nunique().eq(len(assignments)).all():
        raise ValueError('Seed ensemble test patient alignment mismatch.')
    ensemble=combined.groupby('patient_id',as_index=False).agg(y_true=('y_true','max'),y_prob=('y_prob','mean'))
    ensemble.to_csv(Path(folder)/'seed_ensemble_patient_predictions.csv',index=False)
    atomic_json(Path(folder)/'seed_ensemble_patient_metrics.json', metrics(ensemble.y_true,ensemble.y_prob,.5))
    return out
