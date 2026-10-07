# Rules for the coding agent
- Read BRIEF.md first. Follow the "Build order" section (section 13).
- Do not read, search or modify the `venv/` folder, `data/`, `models/` or any video file.
- Do not run video processing or install packages yourself. Give me the PowerShell command and I will run it and paste the output.
- Work on one step at a time. Keep answers short. Do not rewrite files that already work.
- After each step, update PROGRESS.md with: what is done, what is next, how to run it.
- Use synthetic data only. Never ask for real photos or videos.
- Write commands for Windows PowerShell. Explain any new dependency before adding it.

## Build order
A. simulate.py + presence.py + dashboard reconciliation on simulated data (no face models needed)
B. M1 enrolment
C. M2 identity on the sample video (demo mode)
D. M5 evaluation + README
