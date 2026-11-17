#!/bin/bash
# Build the Moodle bundle from the committed state of the repository:
#   code/ (all tracked files), results/ (tables, csv, figures), presentation/ (slides), report/ (PDF and source)
# Output: submission/blur-robust-3dgs-submission.zip. Refuses to run with uncommitted changes, so the bundle matches GitHub.
set -e
cd "$(dirname "$0")/.."

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "uncommitted changes: commit them first so the bundle matches the repository"
    exit 1
fi
for f in docs/report.pdf docs/slides.pptx docs/results.md docs/all_results.csv; do
    [ -f "$f" ] || { echo "missing $f (run: python scripts/build_report_pdf.py, scripts/build_slides.py, scripts/collect.py)"; exit 1; }
done

bundle=submission/bundle
rm -rf submission
mkdir -p $bundle/code $bundle/results $bundle/presentation $bundle/report
git archive HEAD | tar -x -C $bundle/code
cp docs/results.md docs/all_results.csv $bundle/results/
cp -r docs/figures $bundle/results/figures
cp docs/slides.pptx $bundle/presentation/
cp docs/report.pdf docs/report.md $bundle/report/

cat > $bundle/README.txt <<TXT
Blur-Robust 3D Gaussian Splatting via Joint Per-View Blur-Kernel Estimation (CS 663 course project)
Repository: https://github.com/IshaanM05/blur-robust-3dgs   Commit: $(git rev-parse --short HEAD)

code/          the full source (see code/README.md for setup and how to reproduce)
results/       results.md (every table), all_results.csv (one row per run), figures/
presentation/  slides.pptx (10-minute talk, speaker notes included)
report/        report.pdf (paper-style write-up) and report.md (source)
TXT

(cd $bundle && zip -qr ../blur-robust-3dgs-submission.zip .)
echo "built submission/blur-robust-3dgs-submission.zip ($(du -h submission/blur-robust-3dgs-submission.zip | cut -f1)), commit $(git rev-parse --short HEAD)"
