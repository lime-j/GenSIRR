from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

BASE_DIR = Path(__file__).resolve().parent
DATASET_ALIASES = {
    "wild": "sir2",
    "postcard": "sir2",
    "object": "sir2",
}


def load_user_file(path: Path):
    entries = []
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                sample_id, label = line.split(",", 1)
            except ValueError as exc:  # pragma: no cover - sanity guard
                raise ValueError(f"Malformed line in {path}: {line!r}") from exc
            entries.append((sample_id, label))
    if not entries:
        raise ValueError(f"No entries found in {path}")
    return entries


def dataset_from_filename(path: Path, method_name: str) -> tuple[str, str, str]:
    stem = path.stem
    prefix = f"{method_name}_"
    remainder = stem[len(prefix) :] if stem.startswith(prefix) else stem
    marker = "_userstudy"
    if marker not in remainder:
        raise ValueError(f"Unable to parse dataset name from {path.name}")
    dataset_raw, suffix = remainder.split(marker, 1)
    if not dataset_raw:
        raise ValueError(f"Empty dataset segment in {path.name}")
    dataset = DATASET_ALIASES.get(dataset_raw, dataset_raw)
    normalized_user = f"{method_name}_{dataset}{marker}{suffix}"
    return dataset, normalized_user, dataset_raw


def aggregate_user_files(file_infos: list[dict]):
    if not file_infos:
        return None

    sample_votes = defaultdict(list)
    for info in file_infos:
        sample_prefix = info.get("sample_prefix", "")
        for sample_id, label in info["entries"]:
            key = f"{sample_prefix}{sample_id}"
            sample_votes[key].append(label == "success")

    grouped_infos: dict[str, list[dict]] = defaultdict(list)
    for info in file_infos:
        grouped_infos[info["user"]].append(info)

    per_user = []
    failure_rate_lists: dict[str, list[float]] = {}

    for user_index, user in enumerate(sorted(grouped_infos)):
        combined_entries = []
        for info in grouped_infos[user]:
            combined_entries.extend(info["entries"])

        labels = [label for _, label in combined_entries]
        counts = Counter(labels)
        total = len(combined_entries)
        success_rate = counts.get("success", 0) / total
        failure_rates = {
            name: counts[name] / total
            for name in counts
            if name != "success"
        }

        per_user.append(
            {
                "user": user,
                "total": total,
                "success_rate": success_rate,
                "failure_rates": failure_rates,
            }
        )

        for failure in failure_rates:
            if failure not in failure_rate_lists:
                failure_rate_lists[failure] = [0.0] * user_index

        for failure in failure_rate_lists:
            rate = failure_rates.get(failure, 0.0)
            failure_rate_lists[failure].append(rate)

    num_samples = len(sample_votes)
    if num_samples == 0:
        raise ValueError("No samples collected from user files.")

    pass_any = sum(any(votes) for votes in sample_votes.values()) / num_samples
    pass_all = sum(all(votes) for votes in sample_votes.values()) / num_samples
    avg_success = mean(user["success_rate"] for user in per_user)

    avg_failure_rates = {
        failure: mean(rates)
        for failure, rates in failure_rate_lists.items()
    }

    return {
        "per_user": per_user,
        "avg_success": avg_success,
        "pass_any": pass_any,
        "pass_all": pass_all,
        "avg_failure_rates": avg_failure_rates,
        "num_samples": num_samples,
    }


def analyze_method(method_dir: Path):
    user_files = sorted(method_dir.glob("*_user*.txt"))
    if not user_files:
        return None

    dataset_groups: dict[str, list[dict]] = defaultdict(list)
    overall_infos = []

    for path in user_files:
        entries = load_user_file(path)
        dataset, canonical_user, dataset_raw = dataset_from_filename(path, method_dir.name)
        dataset_groups[dataset].append(
            {
                "user": canonical_user,
                "entries": entries,
                "sample_prefix": f"{dataset_raw}/",
            }
        )
        overall_infos.append(
            {
                "user": canonical_user,
                "entries": entries,
                "sample_prefix": f"{dataset_raw}/",
            }
        )

    overall = aggregate_user_files(overall_infos)
    dataset_stats = {
        dataset: aggregate_user_files(infos)
        for dataset, infos in sorted(dataset_groups.items())
    }

    return {
        "method": method_dir.name,
        "overall": overall,
        "datasets": dataset_stats,
    }


def main():
    method_dirs = [d for d in sorted(BASE_DIR.iterdir()) if d.is_dir()]
    analyses = []
    for method_dir in method_dirs:
        analysis = analyze_method(method_dir)
        if analysis is not None:
            analyses.append(analysis)

    if not analyses:
        print("No user study data found.")
        return

    for analysis in analyses:
        print(f"=== {analysis['method']} ===")
        print("Overall:")
        print_stats_block(analysis["overall"], indent="  ")
        if analysis["datasets"]:
            print("Datasets:")
            for dataset, stats in analysis["datasets"].items():
                print(f"  -- {dataset} --")
                print_stats_block(stats, indent="    ")
        print()


def print_stats_block(stats: dict, indent: str = ""):
    for user_info in stats["per_user"]:
        success_pct = user_info["success_rate"] * 100
        print(f"{indent}{user_info['user']}: success {success_pct:.1f}% ({user_info['total']} samples)")
    print(f"{indent}Average success rate: {stats['avg_success'] * 100:.1f}%")
    print(f"{indent}Pass if any user succeeds: {stats['pass_any'] * 100:.1f}%")
    print(f"{indent}Pass if all users succeed: {stats['pass_all'] * 100:.1f}%")
    if stats["avg_failure_rates"]:
        print(f"{indent}Average failure rate by type:")
        for failure, rate in sorted(stats["avg_failure_rates"].items(), key=lambda item: -item[1]):
            print(f"{indent}  {failure}: {rate * 100:.1f}%")


if __name__ == "__main__":
    main()
