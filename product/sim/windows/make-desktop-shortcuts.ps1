# Puts two shortcuts on your desktop: "Teach Pendant (robot)" and "Teach Pendant (twin)".
# Run once from PowerShell:  powershell -ExecutionPolicy Bypass -File C:\dev\cobot-teaching\product\sim\windows\make-desktop-shortcuts.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
foreach ($s in @(@("Teach Pendant (robot)", "Teach-Pendant-Robot.bat"), @("Teach Pendant (twin)", "Teach-Pendant-Twin.bat"))) {
    $lnk = $shell.CreateShortcut((Join-Path $desktop ($s[0] + ".lnk")))
    $lnk.TargetPath = Join-Path $here $s[1]
    $lnk.WorkingDirectory = Split-Path -Parent $here
    $lnk.IconLocation = "$env:SystemRoot\System32\shell32.dll,12"
    $lnk.Save()
    Write-Host "Made: $($s[0]) on the desktop"
}
