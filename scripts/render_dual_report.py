"""Render private, self-contained charts from committed dual-track receipts."""
from __future__ import annotations

import argparse
import base64
import html
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from histology_data.dual_common import ArtifactStore
from scripts._dual_report_data import (
    REPS,
    _write_private,
    open_store,
    prepare_report_data,
    read,
    representative_csv,
    select_representatives,
)


def _save(fig: Any, output: Path, name: str, images: list[Path]) -> None:
    fig.savefig(output / f"{name}.png", dpi=140, bbox_inches="tight",
                metadata={"Software": "dual report postprocessor"})
    fig.savefig(output / f"{name}.svg", bbox_inches="tight",
                metadata={"Creator": "dual report postprocessor"})
    import matplotlib.pyplot as plt
    plt.close(fig)
    images.append(output / f"{name}.png")


def _concentration_rows(representatives: Mapping[str, Any]) -> list[tuple[int, str, int, int, int, float, float, int]]:
    result = []
    for index, (key, _) in enumerate(REPS):
        item = representatives.get(key)
        if item is None:
            continue
        clusters: dict[int, list[dict[str, Any]]] = {}
        for point in item["points"]:
            clusters.setdefault(point["cluster_id"], []).append(point)
        for cluster, rows in sorted(clusters.items()):
            patients = [row["patient_token"] for row in rows]
            photos = [row["photo_token"] for row in rows]
            result.append((index, item["title"], cluster,
                           sum(row["target"] == 0 for row in rows),
                           sum(row["target"] == 1 for row in rows),
                           max(patients.count(value) for value in set(patients)) / len(rows),
                           max(photos.count(value) for value in set(photos)) / len(rows), len(rows)))
    return result


def _atlas_plots(output: Path, atlas: Mapping[str, Any], images: list[Path]) -> None:
    for lens, data in atlas.get("lenses", {}).items():
        representatives = data["representatives"]
        if representatives:
            fig, axes = _new_figure(2, 3, (15, 9))
            for ax, (key, title) in zip(axes.flat, REPS):
                item = representatives.get(key)
                if item is None:
                    ax.axis("off")
                    continue
                points = item["points"]
                colors = ["#bdbdbd" if row["cluster_id"] == -1 else
                          _cluster_color(row["cluster_id"]) for row in points]
                ax.scatter([row["x"] for row in points], [row["y"] for row in points],
                           c=colors, s=12, alpha=.8)
                noise = item["metrics"]["partition"]["noise_fraction"]
                ax.set(title=f"{title}; noise {noise:.1%}",
                       xlabel="train PCA32 component 1", ylabel="train PCA32 component 2")
            fig.suptitle(f"A0 five-patient development atlas · {lens}x · IDs local to each panel")
            _save(fig, output, f"a0_scatter_lens_{lens}", images)

        configs = data["configs"]
        hdb = [row for row in configs if row["method"] in {"eom", "leaf"}]
        if hdb:
            fig, axes = _new_figure(1, 2, (15, 5))
            settings = sorted({(row["min_cluster_size"], row["min_samples"]) for row in hdb})
            for method, marker, color in (("eom", "o", "#4c78a8"), ("leaf", "s", "#f58518")):
                rows = [next(row for row in hdb if (row["min_cluster_size"], row["min_samples"]) == setting
                             and row["method"] == method) for setting in settings]
                axes[0].plot(range(len(settings)), [row["coverage"] for row in rows], marker=marker,
                             color=color, label=method.upper())
            axes[0].set(xticks=range(len(settings)), xticklabels=[f"{size}/{sample}" for size, sample in settings],
                        ylim=(0, 1.02), ylabel="assigned fraction", xlabel="min cluster size / min samples",
                        title="HDBSCAN EOM / leaf coverage")
            axes[0].tick_params(axis="x", labelrotation=45)
            axes[0].legend()
            labels = [f"{row['method']} {row.get('min_cluster_size', row.get('k'))}"
                      for row in configs]
            axes[1].bar(range(len(configs)), [row["clusters"] for row in configs],
                        color=["#4c78a8" if row["method"] == "eom" else
                               "#f58518" if row["method"] == "leaf" else
                               "#54a24b" if row["method"] == "ward" else "#b279a2" for row in configs])
            axes[1].set(xticks=range(len(configs)), xticklabels=labels, ylabel="clusters / requested K",
                        title="HDBSCAN, Ward and KMeans controls")
            axes[1].tick_params(axis="x", labelrotation=70, labelsize=6)
            _save(fig, output, f"a0_grid_lens_{lens}", images)

        concentrations = _concentration_rows(representatives)
        if concentrations:
            fig, axes = _new_figure(1, 2, (15, max(6, .20 * len(concentrations))))
            y = range(len(concentrations))
            negative = [row[3] for row in concentrations]
            positive = [row[4] for row in concentrations]
            axes[0].barh(y, negative, color="#4c78a8", label="source case label 0")
            axes[0].barh(y, positive, left=negative, color="#e45756", label="source case label 1")
            axes[0].set(yticks=list(y), yticklabels=[f"{row[1]} / {row[2]}" for row in concentrations],
                        xlabel="sampled tile rows", title="Source case-label counts (not patch truth)")
            axes[0].tick_params(axis="y", labelsize=6)
            axes[0].legend(fontsize=7)
            for index, (_, title) in enumerate(REPS):
                rows = [row for row in concentrations if row[0] == index]
                if rows:
                    axes[1].scatter([row[5] for row in rows], [row[6] for row in rows],
                                    s=[max(12, min(120, row[7])) for row in rows],
                                    label=title, alpha=.8)
            axes[1].plot([0, 1], [0, 1], color="#aaa", linewidth=.7)
            axes[1].set(xlim=(0, 1.03), ylim=(0, 1.03), xlabel="largest patient share",
                        ylabel="largest photo share", title="Patient / photo concentration")
            axes[1].legend(fontsize=6)
            _save(fig, output, f"a0_counts_concentration_lens_{lens}", images)


def _cluster_color(cluster: int) -> Any:
    import matplotlib.pyplot as plt
    return plt.get_cmap("tab20")(cluster % 20)


def _new_figure(rows: int, columns: int, size: tuple[float, float]) -> tuple[Any, Any]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt.subplots(rows, columns, figsize=size, constrained_layout=True)


def _oof_plots(output: Path, models: Mapping[str, list[dict[str, Any]]], matched: list[str],
               a_by: Mapping[str, Any], b_by: Mapping[str, Any], images: list[Path]) -> None:
    model_rows = [row for rows in models.values() for row in rows]
    if model_rows:
        fig, ax = _new_figure(1, 1, (max(10, .5 * len({r["case_token"] for r in model_rows})), 6))
        cases = sorted({row["case_token"] for row in model_rows})
        position = {case: index for index, case in enumerate(cases)}
        palette = [("A1", "#4c78a8", "o"), ("A counts", "#72b7b2", "s"),
                   ("B0", "#e45756", "^"), ("B counts", "#f2cf5b", "D")]
        palette.extend((name, f"C{index % 10}", "P") for index, name in enumerate(
            key for key in models if key.startswith("B1 ")))
        for offset, (name, color, marker) in enumerate(palette):
            for index, row in enumerate(models.get(name, [])):
                x = position[row["case_token"]] + (offset - (len(palette) - 1) / 2) * .09
                ax.scatter(x, row["probability"], color=color, marker=marker,
                           edgecolor="black" if row["target"] else "white", linewidth=.5,
                           label=name if index == 0 else None)
        ax.axhline(.5, color="#555", linewidth=.8, linestyle="--")
        ax.set(xticks=range(len(cases)), xticklabels=cases, ylim=(-.03, 1.03),
               ylabel="OOF probability", xlabel="opaque case token",
               title="A1 / B0 OOF probabilities and counts-only controls")
        ax.tick_params(axis="x", labelrotation=70, labelsize=7)
        ax.legend(ncol=2)
        _save(fig, output, "oof_case_scatter", images)
    if matched:
        fig, ax = _new_figure(1, 1, (7, 7))
        for target, color, marker in ((0, "#4c78a8", "o"), (1, "#e45756", "^")):
            rows = [case for case in matched if a_by[case]["target"] == target]
            ax.scatter([a_by[case]["probability"] for case in rows],
                       [b_by[case]["probability"] for case in rows], color=color, marker=marker,
                       label=f"source case label {target}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#777", linewidth=.8)
        ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel="A1 probability", ylabel="B0 probability",
               title=f"Matched A1 / B0 OOF cases · n={len(matched)}")
        ax.legend()
        _save(fig, output, "oof_a1_b0_matched", images)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    probability = np.clip(np.asarray([row["probability"] for row in rows], dtype=np.float64), 1e-15, 1 - 1e-15)
    return {"n": len(rows), "negative": int((target == 0).sum()), "positive": int((target == 1).sum()),
            "mean_p0": float(probability[target == 0].mean()) if np.any(target == 0) else "",
            "mean_p1": float(probability[target == 1].mean()) if np.any(target == 1) else "",
            "log_loss": float(-np.mean(target * np.log(probability) + (1 - target) * np.log(1 - probability))),
            "brier": float(np.mean((probability - target) ** 2))}


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def cells(values: list[Any], tag: str) -> str:
        return "".join(f"<{tag}>{html.escape(str(value))}</{tag}>" for value in values)
    return "<table><thead><tr>" + cells(headers, "th") + "</tr></thead><tbody>" + "".join(
        "<tr>" + cells(row, "td") + "</tr>" for row in rows
    ) + "</tbody></table>"


def _write_reports(output: Path, manifest: Mapping[str, Any], images: list[Path],
                   models: Mapping[str, list[dict[str, Any]]]) -> None:
    summaries = [[name, *_summary(rows).values()] for name, rows in models.items() if rows]
    headers = ["Model", "n", "negative", "positive", "mean P0", "mean P1", "log loss", "Brier"]
    status = (f"Report **{manifest['status']}** · A0 **{manifest['a0']['status']}** · "
              f"A/B **{manifest['matched_oof']['status']}** · B1 **{manifest['b1']['status']}**")
    md = ["# Dual-track exploratory report", "", status,
          "A0 uses the five-patient outer_0/inner_0 training scope. Source case labels describe the origin case, not patch-level truth.",
          "", "## Case-level OOF summary", "", "| " + " | ".join(headers) + " |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    md.extend("| " + " | ".join(map(str, row)) + " |" for row in summaries)
    sections = [f"<h1>Dual-track exploratory report</h1><p>{html.escape(status)}</p>",
                "<p>A0 uses the five-patient outer_0/inner_0 training scope. Source case labels describe the origin case, not patch-level truth.</p>",
                "<h2>Case-level OOF summary</h2>", _table(headers, summaries)]
    configs = []
    for lens, data in manifest["a0"].get("lenses", {}).items():
        configs.extend([[lens, row["method"],
                         f"{row.get('min_cluster_size', '')}/{row.get('min_samples', '')}/k{row.get('k', '')}",
                         f"{row['coverage']:.3f}", f"{row['noise']:.3f}", row["clusters"]]
                        for row in data["configs"]])
    config_headers = ["lens", "method", "setting", "coverage", "noise", "clusters"]
    md.extend(("", "## A0 development configurations", "", "| " + " | ".join(config_headers) + " |",
               "|---|---|---|---:|---:|---:|"))
    md.extend("| " + " | ".join(map(str, row)) + " |" for row in configs)
    sections.extend(("<h2>A0 development configurations</h2>", _table(config_headers, configs)))
    for image in images:
        md.extend(("", f"![{image.name}]({image.name})"))
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        sections.append(f'<figure><img alt="{html.escape(image.name)}" src="data:image/png;base64,{encoded}"></figure>')
    (output / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    css = "<style>body{font:15px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}"
    css += "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.35rem}img{max-width:100%}</style>"
    (output / "report.html").write_text("<!doctype html><meta charset='utf-8'><title>Dual report</title>" +
                                         css + "<body>" + "".join(sections) + "</body>", encoding="utf-8")


def render_dual_report(a: ArtifactStore, b: ArtifactStore, shared: ArtifactStore, source: Mapping[str, Any],
                       output_dir: Path, private_map: Path, *, b1: ArtifactStore | None = None,
                       b1_artifacts: Mapping[str, str] | None = None,
                       representatives_private_path: Path | None = None) -> dict[str, Any]:
    """Render committed A0/A1/B0 receipts and privately export up to 96 review refs.

    The raw representative CSV defaults to a sibling of ``private_map`` named
    ``<private_map_stem>.representatives.csv``. Keep that file with the private
    map; it contains source tile and feature-part identifiers for later retrieval.
    """
    output = Path(output_dir).expanduser().absolute()
    if output.is_symlink():
        raise ValueError("Report output directory cannot be a symlink.")
    rep_path = (Path(representatives_private_path).expanduser().absolute() if representatives_private_path else
                Path(private_map).expanduser().absolute().with_name(
                    Path(private_map).stem + ".representatives.csv"))
    if (rep_path.is_symlink() or rep_path == Path(private_map).expanduser().absolute()
            or rep_path.resolve().is_relative_to(output.resolve())):
        raise ValueError("Representative CSV must be a distinct private file outside the report directory.")
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_report_data(a, b, shared, source, output, private_map, b1=b1, b1_artifacts=b1_artifacts)
    images: list[Path] = []
    _atlas_plots(output, data["atlas"], images)
    _oof_plots(output, data["models"], data["matched"], data["a_by"], data["b_by"], images)
    (output / "point_lineage.csv").write_text(data["point_csv"], encoding="utf-8")
    (output / "oof_lineage.csv").write_text(data["oof_csv"], encoding="utf-8")
    public_manifest = data["manifest"]
    public_manifest["files"] = ([path.name for path in images]
                                 + [path.with_suffix(".svg").name for path in images]
                                 + ["point_lineage.csv", "oof_lineage.csv", "report_manifest.json",
                                    "report.md", "report.html"])
    (output / "report_manifest.json").write_text(
        json.dumps(public_manifest, sort_keys=True, allow_nan=False, indent=2), encoding="utf-8")
    _write_reports(output, public_manifest, images, data["models"])
    _write_private(rep_path, representative_csv(select_representatives(data["atlas"])).encode("utf-8"))
    return public_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a-store", "b-store", "shared-store", "output-dir", "private-mapping"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--representatives-private", type=Path)
    parser.add_argument("--b1-store", type=Path)
    parser.add_argument("--b1-artifact", action="append", default=[], metavar="CANDIDATE=NAME")
    args = parser.parse_args()
    names = {}
    for item in args.b1_artifact:
        candidate, separator, artifact = item.partition("=")
        if not separator or candidate in names or not candidate or not artifact:
            parser.error("--b1-artifact must be unique CANDIDATE=NAME")
        names[candidate] = artifact
    shared = open_store(args.shared_store)
    source = read(shared, "source_photo_map.pkl", pickle.loads, required=True)
    result = render_dual_report(
        open_store(args.a_store), open_store(args.b_store), shared, source, args.output_dir,
        args.private_mapping, b1=open_store(args.b1_store) if args.b1_store else None,
        b1_artifacts=names or None, representatives_private_path=args.representatives_private,
    )
    print(json.dumps({"status": result["status"], "a0": result["a0"]["status"],
                      "matched_oof": result["matched_oof"]["status"], "b1": result["b1"]["status"]}))


if __name__ == "__main__":
    main()
