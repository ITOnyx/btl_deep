# Tạo shortcut "Tra cứu thời trang" trên Desktop (và Start Menu) chạy app/desktop.py bằng pythonw (không hiện console).
# Chạy:  powershell -ExecutionPolicy Bypass -File scripts\create_desktop_shortcut.ps1
param(
    [string]$Pythonw = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not $Pythonw) {
    $py = (Get-Command python -ErrorAction SilentlyContinue | Where-Object { $_.Source -notlike "*WindowsApps*" } | Select-Object -First 1).Source
    if (-not $py) { throw "Không tìm thấy python.exe (bản cài thật, không phải alias WindowsApps). Truyền -Pythonw <đường dẫn>." }
    $Pythonw = Join-Path (Split-Path $py) "pythonw.exe"
}
if (-not (Test-Path $Pythonw)) { throw "Không tìm thấy $Pythonw" }

$icon = Join-Path $root "app\assets\app.ico"
if (-not (Test-Path $icon)) { & (Join-Path (Split-Path $Pythonw) "python.exe") (Join-Path $root "scripts\make_app_icon.py") | Out-Null }

$shell = New-Object -ComObject WScript.Shell
$targets = @(
    (Join-Path ([Environment]::GetFolderPath("Desktop")) "Tra cứu thời trang.lnk"),
    (Join-Path ([Environment]::GetFolderPath("Programs")) "Tra cứu thời trang.lnk")
)
foreach ($lnkPath in $targets) {
    # WScript.Shell chỉ lưu được tên tệp ANSI: lưu tên tạm không dấu rồi đổi sang tên có dấu
    $tmp = Join-Path (Split-Path $lnkPath) ("fashion_search_tmp_" + [guid]::NewGuid().ToString("N") + ".lnk")
    $lnk = $shell.CreateShortcut($tmp)
    $lnk.TargetPath = $Pythonw
    $lnk.Arguments = "`"$(Join-Path $root 'app\desktop.py')`""
    $lnk.WorkingDirectory = $root
    $lnk.IconLocation = "$icon,0"
    $lnk.Description = "Tra cuu san pham thoi trang (SigLIP)"
    $lnk.Save()
    if (Test-Path -LiteralPath $lnkPath) { Remove-Item -LiteralPath $lnkPath -Force }
    Move-Item -LiteralPath $tmp -Destination $lnkPath
    Write-Output "Đã tạo: $lnkPath"
}
