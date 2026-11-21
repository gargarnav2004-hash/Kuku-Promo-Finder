#!/bin/bash
# Builds submission/code.zip and submission/raw_outputs.zip
cd "$(dirname "$0")"
rm -rf submission && mkdir -p submission
zip -q submission/code.zip app.py llm.py parse.py pipeline.py run.py config.yaml requirements.txt .env.example package.sh \
  README.md NOTES_FOR_DOC.md VIDEO_SCRIPT.md INTERVIEW_PREP.md
zip -qr submission/raw_outputs.zip outputs
unzip -l submission/code.zip; unzip -l submission/raw_outputs.zip
