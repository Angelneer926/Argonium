#!/usr/bin/env python3
import subprocess
import time

# 你的 MC 题库
QUESTIONS_FILE = "questions_oss120_epimem.json"

# 当前确实可用的模型
MODELS = ["llama70", "oss120"]

def run_one_pair(test_model, grader):
    """
    让 grader 模型给 test_model 的回答打分
    返回 accuracy 百分比
    """
    cmd = [
        "python", "argonium_score_parallel_v9.py",
        QUESTIONS_FILE,
        "--model", test_model,
        "--grader", grader,
        "--format", "mc",
        "--parallel", "4",
        "--seed", "42",
    ]
    # 如果你只想抽样，可以加 "--random", "120" 之类
    # 不加就是用整个 questions_oss120_epimem.json

    print(f"\n=== Running: test_model={test_model}, grader={grader} ===")
    print("Command:", " ".join(cmd))

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()

    if result.returncode != 0:
        print(f"Run failed for pair ({test_model}, {grader})")
        print("stderr:", result.stderr[:500])
        return None

    stdout = result.stdout
    print(stdout)  # 方便之后截图

    acc = extract_accuracy(stdout)
    if acc is None:
        print("Warning: could not parse accuracy from output")
    else:
        print(f"Parsed accuracy: {acc:.2f}% in {end - start:.1f}s")
    return acc

def extract_accuracy(stdout: str):
    """
    从 argonium_score_parallel_v9.py 的输出里解析 overall accuracy 百分比
    """
    for line in stdout.splitlines():
        line = line.strip()
        if "Overall accuracy:" in line:
            # 例如: Overall accuracy: 88.00% (44.0/50)
            try:
                part = line.split("Overall accuracy:")[1]
                percent_str = part.split("%")[0].strip()
                return float(percent_str)
            except Exception:
                continue
        if "Multiple-choice questions:" in line and "accuracy" in line:
            # 例如: Multiple-choice questions: 88.00% accuracy (44/50)
            try:
                part = line.split("%")[0]
                percent_str = part.split()[-1]
                return float(percent_str)
            except Exception:
                continue
    return None

def main():
    # results[grader][test_model] = accuracy
    results = {g: {m: None for m in MODELS} for g in MODELS}

    for grader in MODELS:
        for test_model in MODELS:
            acc = run_one_pair(test_model, grader)
            results[grader][test_model] = acc

    print("\n==========================================")
    print("Cross grading accuracy table (rows = test model, columns = grader)")
    print("==========================================\n")

    # 打印 TSV 表头
    header = ["model"] + [f"graded_by_{g}" for g in MODELS]
    print("\t".join(header))

    for m in MODELS:
        row = [m]
        for g in MODELS:
            acc = results[g][m]
            if acc is None:
                row.append("N/A")
            else:
                row.append(f"{acc:.1f}")
        print("\t".join(row))

    # 打印 markdown 表格，方便直接拷贝进报告
    print("\nMarkdown table:\n")
    print("| Model | " + " | ".join([f"Graded by {g}" for g in MODELS]) + " |")
    print("|" + " --- |" * (len(MODELS) + 1))
    for m in MODELS:
        cells = []
        for g in MODELS:
            acc = results[g][m]
            if acc is None:
                cells.append("N/A")
            else:
                cells.append(f"{acc:.1f}%")
        print("| " + m + " | " + " | ".join(cells) + " |")

if __name__ == "__main__":
    main()
