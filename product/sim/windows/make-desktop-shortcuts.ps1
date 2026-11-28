# Puts two shortcuts on your desktop: "Teach Pendant (robot)" and "Teach Pendant (twin)".
# Run once from PowerShell:  powershell -ExecutionPolicy Bypass -File C:\dev\cobot-teaching\product\sim\windows\make-desktop-shortcuts.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
foreach ($s in @(@("Teach Pendant (robot)", "Teach-Pendant-Robot.bat", "teach-pendant-robot.ico"), @("Teach Pendant (twin)", "Teach-Pendant-Twin.bat", "teach-pendant-twin.ico"))) {
    $lnk = $shell.CreateShortcut((Join-Path $desktop ($s[0] + ".lnk")))
    $lnk.TargetPath = Join-Path $here $s[1]
    $lnk.WorkingDirectory = Split-Path -Parent $here
    $lnk.IconLocation = (Join-Path $here $s[2]) + ",0"
    $lnk.Save()
    Write-Host "Made: $($s[0]) on the desktop"
}
