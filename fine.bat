@echo off
echo Starting per-file commits...

for /r %%f in (*) do (
    echo Committing %%f
    git add "%%f"
    git commit -m "Add %%f"
    git push
)

echo Done!
pause
