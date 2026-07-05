# Prod watchdog: keep the bot alive across crashes (network blips, proxy
# restarts). Run from the project root:  powershell -File scripts\run_bot.ps1
# Stop with Ctrl+C (the loop exits only on manual interrupt).
while ($true) {
    Write-Host "[watchdog] starting bot $(Get-Date -Format s)"
    python -m src.bot
    Write-Host "[watchdog] bot exited (code $LASTEXITCODE), restart in 5s"
    Start-Sleep -Seconds 5
}
