; 暖暖记忆助手 - Inno Setup 安装脚本
; 生成单文件安装程序：暖暖记忆助手-Setup.exe
;
; 使用方式（需先安装 Inno Setup 6）：
;   ISCC.exe 安装包.iss
;
; 或在 Inno Setup Studio 中打开此文件按 F9 编译。

#define MyAppName "暖暖记忆助手"
; 版本号由 build_exe.py 通过环境变量 APP_VERSION 传入
#define APP_VERSION GetEnv("APP_VERSION")
#if APP_VERSION == ""
  #define APP_VERSION "1.0.0"
#endif
#define MyAppVersion APP_VERSION
#define MyAppPublisher "暖暖记忆助手"
#define MyAppExeName "暖暖记忆助手.exe"
#define MyAppSourceDir "dist\暖暖记忆助手"
#define MyAppIcon "appicon.ico"

[Setup]
AppId={{B7F3A2C1-9D4E-4F8A-9B6C-1E2D3F4A5B6C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=暖暖记忆助手-Setup
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
SetupIconFile={#MyAppIcon}
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
; 安装界面使用默认英文（程序本身为中文，不影响使用）
[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "在桌面创建快捷方式"; GroupDescription: "附加图标:"; Flags: checkedonce
Name: "startup"; Description: "开机自启动"; GroupDescription: "附加图标:"; Flags: unchecked

[Files]
; 打包整个 dist/暖暖记忆助手/ 目录
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; 图标文件安装到程序目录（exe 运行时需要）
Source: "{#MyAppIcon}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIcon}"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIcon}"; Tasks: desktopicon
Name: "{autostartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIcon}"; Tasks: startup

[Run]
; 安装完成后可选择立即启动
Filename: "{app}\{#MyAppExeName}"; Description: "立即启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 卸载时清理程序文件（保留用户数据 %APPDATA%/记忆助手/）
Type: filesandordirs; Name: "{app}"

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
