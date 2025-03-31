!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"
!include "WordFunc.nsh"

; Define the name of the installer
OutFile "..\dist\FreeScribeInstaller.exe"

; Silent mode flags:
; /S - Silent mode
; /ARCH=[CPU|NVIDIA] - Force architecture selection
; /K - Kill running instance before installation

; Define the default installation directory to AppData
InstallDir "$PROGRAMFILES\FreeScribe"

; Define the name of the installer
Name "FreeScribe"

; Define the version of the installer
VIProductVersion "0.0.0.1"
VIAddVersionKey "ProductName" "FreeScribe"
VIAddVersionKey "FileVersion" "0.0.0.1"
VIAddVersionKey "LegalCopyright" "Copyright (c) 2023-2024 Braedon Hendy"
VIAddVersionKey "FileDescription" "FreeScribe Installer"

; Define the logo image
!define MUI_ICON ./assets/logo.ico
!define MIN_CUDA_DRIVER_VERSION 527.41 ; The nvidia graphic driver that is compatiable with Cuda 12.1

; Variables for checkboxes
Var /GLOBAL CPU_RADIO
Var /GLOBAL NVIDIA_RADIO
Var /GLOBAL SELECTED_OPTION
Var /GLOBAL REMOVE_CONFIG_CHECKBOX
Var /GLOBAL REMOVE_CONFIG
Var /GLOBAL Got_Running_Instance

!macro CheckRunningInstanceMacro
    nsExec::ExecToStack 'cmd /c tasklist /FI "IMAGENAME eq freescribe-client.exe" /NH | find /I "freescribe-client.exe" > nul'
    Pop $0 ; Return value
    ${If} $0 == 0
        StrCpy $Got_Running_Instance "1"
    ${Else}
        StrCpy $Got_Running_Instance "0"
    ${EndIf}
!macroend

!macro HideNextButtonMacro
    GetDlgItem $R0 $HWNDPARENT 1 ; Get the handle of the "Next" button
    ShowWindow $R0 ${SW_HIDE}    ; Hide the "Next" button
!macroend

!macro ShowNextButtonMacro
    GetDlgItem $R0 $HWNDPARENT 1 ; Get the handle of the "Next" button
    ShowWindow $R0 ${SW_SHOW}    ; Show the "Next" button
!macroend

!macro GotoNextPageMacro
    GetDlgItem $1 $HWNDPARENT 1 ; Get the "Next" button handle
    SendMessage $HWNDPARENT ${WM_COMMAND} 1 $1 ; Simulate clicking the "Next" button
!macroend

!macro HideBackButtonMacro
    GetDlgItem $R0 $HWNDPARENT 3 ; Get the handle of the "Back" button
    ShowWindow $R0 ${SW_HIDE}    ; Hide the "Back" button
!macroend

Function HideNextButton
    !insertmacro HideNextButtonMacro
FunctionEnd

Function ShowNextButton
    !insertmacro ShowNextButtonMacro
FunctionEnd

Function GotoNextPage
    !insertmacro GotoNextPageMacro
FunctionEnd

Function HideBackButton
    !insertmacro HideBackButtonMacro
FunctionEnd

Function un.HideNextButton
    !insertmacro HideNextButtonMacro
FunctionEnd

Function un.ShowNextButton
    !insertmacro ShowNextButtonMacro
FunctionEnd

Function un.HideBackButton
    !insertmacro HideBackButtonMacro
FunctionEnd

Function un.GotoNextPage
    !insertmacro GotoNextPageMacro
FunctionEnd

!macro KillFreeScribeProcessMacro
    nsExec::ExecToStack 'taskkill /F /IM freescribe-client.exe'
    Pop $0 ; Return value

    ${If} $0 == 0
        MessageBox MB_OK "FreeScribe process has been terminated."
        Return
    ${Else}
        MessageBox MB_OK|MB_ICONEXCLAMATION "Failed to terminate FreeScribe process. Please close it manually."
        Return
    ${EndIf}
!macroend

Function KillFreeScribeProcess
    !insertmacro KillFreeScribeProcessMacro
FunctionEnd

Function un.KillFreeScribeProcess
    !insertmacro KillFreeScribeProcessMacro
FunctionEnd

Function Check_For_Old_Version_In_App_Data
    ; Check if the old version exists in AppData
    IfFileExists "$APPDATA\FreeScribe\freescribe-client.exe" 0 OldVersionDoesNotExist
        ; Open Dialog to ask user if they want to uninstall the old version
        MessageBox MB_YESNO|MB_ICONQUESTION "An old version of FreeScribe has been detected. Would you like to uninstall it?" IDYES UninstallOldVersion IDNO OldVersionDoesNotExist
        UninstallOldVersion:
            ; Remove the contents/folders of the old version
            RMDir /r "$APPDATA\FreeScribe\_internal"
            RMDir /r "$APPDATA\FreeScribe\models"

            ; Remove the old version executable
            Delete "$APPDATA\FreeScribe\freescribe-client.exe"

            ; Remove the uninstaller entry from the Control Panel
            Delete "$APPDATA\FreeScribe\uninstall.exe"

            ; Remove the start menu shortcut
            Delete "$SMPROGRAMS\FreeScribe\FreeScribe.lnk"
            RMDir "$SMPROGRAMS\FreeScribe"

            ; Show message when uninstallation is complete
            MessageBox MB_OK "FreeScribe has been successfully uninstalled."
    OldVersionDoesNotExist:
FunctionEnd


; Function to create a custom page with CPU/NVIDIA options
Function ARCHITECTURE_SELECT
    Call Check_For_Old_Version_In_App_Data
    !insertmacro MUI_HEADER_TEXT "Architecture Selection" "Choose your preferred installation architecture based on your hardware"

    nsDialogs::Create 1018
    Pop $0

    ${If} $0 == error
        Abort
    ${EndIf}

    ; Main instruction text for architecture selection
    ${NSD_CreateLabel} 0 0 100% 12u "Choose your preferred installation architecture based on your hardware:"
    Pop $0

    ; Radio button for CPU
    ${NSD_CreateRadioButton} 10 15u 100% 10u "CPU"
    Pop $CPU_RADIO
    ${NSD_Check} $CPU_RADIO
    StrCpy $SELECTED_OPTION "CPU"

    ; CPU explanation text (grey with padding)
    ${NSD_CreateLabel} 20 25u 100% 20u "Recommended for most users. Runs on any modern processor and provides good performance for general use."
    Pop $0
    SetCtlColors $0 808080 transparent

    ; Radio button for NVIDIA
    ${NSD_CreateRadioButton} 10 55u 100% 10u "NVIDIA"
    Pop $NVIDIA_RADIO

    ; NVIDIA explanation text (grey with padding)
    ${NSD_CreateLabel} 20 65u 100% 30u "Choose this option if you have an NVIDIA GPU. Provides accelerated performance. Only select if you have a Nvidia GPU installed."
    Pop $0
    SetCtlColors $0 808080 transparent

    ; Bottom padding (10u of space)
    ${NSD_CreateLabel} 0 95u 100% 10u ""
    Pop $0

    ${NSD_OnClick} $CPU_RADIO OnRadioClick
    ${NSD_OnClick} $NVIDIA_RADIO OnRadioClick

    nsDialogs::Show
FunctionEnd

Function ARCHITECTURE_SELECT_LEAVE
    ${If} $SELECTED_OPTION == "NVIDIA"
        Call CheckNvidiaDrivers
    ${EndIf}
FunctionEnd

; Callback function for radio button clicks
Function OnRadioClick
    Pop $0 ; Get the handle of the clicked control

    ${If} $0 == $CPU_RADIO
        StrCpy $SELECTED_OPTION "CPU"
    ${ElseIf} $0 == $NVIDIA_RADIO
        StrCpy $SELECTED_OPTION "NVIDIA"
    ${EndIf}
FunctionEnd

; Function to show message box on finish
Function .onInstSuccess
    ; Check if silent, if is silent skip message box prompt
    IfSilent +2
    MessageBox MB_OK "Installation completed successfully! Please note upon first launch start time may be slow. Please wait for the program to open!"
FunctionEnd

Function un.onInit
    !insertmacro CheckRunningInstanceMacro
FunctionEnd

; Checks on installer start
Var RunningInstanceDialog
Var ForceStopButton
Var RetryButton

Var StatusLabel

Function un.CreateRunningInstancePage
    ${If} $Got_Running_Instance == "0"
        Abort
    ${EndIf}
    !insertmacro MUI_HEADER_TEXT "Running Instance Detected" ""

    nsDialogs::Create 1018
    Pop $RunningInstanceDialog

    ${If} $RunningInstanceDialog == error
        Abort
    ${EndIf}

    ; Create status label
    ${NSD_CreateLabel} 0 10u 100% 24u "FreeScribe is currently running.$\n$\nPlease choose how to proceed: Force Stop or close it manually and Retry"
    Pop $StatusLabel

    ; Create Force Stop button
    ${NSD_CreateButton} 10% 50u 30% 12u "Force Stop"
    Pop $ForceStopButton
    ${NSD_OnClick} $ForceStopButton un.OnForceStopClick

    ; Create Retry button
    ${NSD_CreateButton} 45% 50u 30% 12u "Retry"
    Pop $RetryButton
    ${NSD_OnClick} $RetryButton un.OnRetryClick

    Call un.HideNextButton
    Call un.HideBackButton

    nsDialogs::Show
FunctionEnd

Function un.OnForceStopClick
    Call un.KillFreeScribeProcess
    nsExec::ExecToStack 'cmd /c tasklist /FI "IMAGENAME eq freescribe-client.exe" /NH | find /I "freescribe-client.exe" > nul'
    Pop $0

    ${If} $0 == 0
        ${NSD_SetText} $StatusLabel "Unable to terminate FreeScribe.$\nPlease close it manually and click Retry."
    ${Else}
        StrCpy $Got_Running_Instance "0"
        Call un.ShowNextButton
        Call un.GotoNextPage
        Abort ; Close the dialog and continue uninstallation
    ${EndIf}
FunctionEnd

Function un.OnRetryClick
    nsExec::ExecToStack 'cmd /c tasklist /FI "IMAGENAME eq freescribe-client.exe" /NH | find /I "freescribe-client.exe" > nul'
    Pop $0

    ${If} $0 == 0
        ${NSD_SetText} $StatusLabel "FreeScribe is still running.$\n$\nPlease choose how to proceed: Force Stop or close it manually and Retry"
    ${Else}
        StrCpy $Got_Running_Instance "0"
        Call un.ShowNextButton
        Call un.GotoNextPage
        Abort ; Close the dialog and continue uninstallation
    ${EndIf}
FunctionEnd

PageEx custom
    PageCallbacks CreateRunningInstancePagePre
PageExEnd

Function CreateRunningInstancePage
    ; Skip this page in silent mode
    IfSilent 0 +2
    Abort
    
    ${If} $Got_Running_Instance == "0"
        Abort
    ${EndIf}
    !insertmacro MUI_HEADER_TEXT "Running Instance Detected" ""

    nsDialogs::Create 1018
    Pop $RunningInstanceDialog

    ${If} $RunningInstanceDialog == error
        Abort
    ${EndIf}

    ; Create status label
    ${NSD_CreateLabel} 0 10u 100% 24u "FreeScribe is currently running.$\n$\nPlease choose how to proceed: Force Stop or close it manually and Retry"
    Pop $StatusLabel

    ; Create Force Stop button
    ${NSD_CreateButton} 10% 50u 30% 12u "Force Stop"
    Pop $ForceStopButton
    ${NSD_OnClick} $ForceStopButton OnForceStopClick

    ; Create Retry button
    ${NSD_CreateButton} 45% 50u 30% 12u "Retry"
    Pop $RetryButton
    ${NSD_OnClick} $RetryButton OnRetryClick

    ${If} $Got_Running_Instance == "1"
        Call HideNextButton
    ${Else}
        Call ShowNextButton
    ${EndIf}
    Call HideBackButton

    nsDialogs::Show
FunctionEnd

Function CreateRunningInstancePagePre
    ${If} $Got_Running_Instance == "0"
        Abort
    ${EndIf}
FunctionEnd

Function OnForceStopClick
    Call KillFreeScribeProcess
    nsExec::ExecToStack 'cmd /c tasklist /FI "IMAGENAME eq freescribe-client.exe" /NH | find /I "freescribe-client.exe" > nul'
    Pop $0

    ${If} $0 == 0
        ${NSD_SetText} $StatusLabel "Unable to terminate FreeScribe.$\nPlease close it manually and click Retry."
    ${Else}
        StrCpy $Got_Running_Instance "0"
        Call ShowNextButton
        Call GotoNextPage
        Abort ; Close the dialog and continue installation
    ${EndIf}
FunctionEnd

Function OnRetryClick
    nsExec::ExecToStack 'cmd /c tasklist /FI "IMAGENAME eq freescribe-client.exe" /NH | find /I "freescribe-client.exe" > nul'
    Pop $0

    ${If} $0 == 0
        ${NSD_SetText} $StatusLabel "FreeScribe is still running.$\n$\nPlease choose how to proceed: Force Stop or close it manually and Retry"
    ${Else}
        StrCpy $Got_Running_Instance "0"
        Call ShowNextButton
        Call GotoNextPage
        Abort ; Close the dialog and continue installation
    ${EndIf}
FunctionEnd

Function .onInit
    !insertmacro CheckRunningInstanceMacro

    IfSilent SILENT_MODE NOT_SILENT_MODE

    SILENT_MODE:
        ${GetParameters} $R0
        ${GetOptions} $R0 "/ARCH=" $R1
        ${If} $R1 != ""
            StrCpy $SELECTED_OPTION $R1
        ${EndIf}
        
        ; Check for /K flag to kill running instance
        ${GetOptions} $R0 "/K" $R2
        ${IfNot} ${Errors}
            Call KillFreeScribeProcess
            !insertmacro CheckRunningInstanceMacro ; Re-check after killing
        ${EndIf}
        
        ; Skip running instance page in silent mode
        StrCpy $Got_Running_Instance "0"
        Return

    NOT_SILENT_MODE:
FunctionEnd

Function CleanUninstall
    ; Remove the contents/folders of the old version
    RMDir /r "$INSTDIR\_internal"

    ; Remove the old version executable
    Delete "$INSTDIR\freescribe-client.exe"

    ; Remove the uninstaller entry from the Control Panel
    Delete "$INSTDIR\uninstall.exe"

    ; Remove the start menu shortcut
    Delete "$SMPROGRAMS\FreeScribe\FreeScribe.lnk"
    RMDir "$SMPROGRAMS\FreeScribe"
FunctionEnd

Function CheckForOldConfig
    ; Check if the old version exists in AppData
    IfFileExists "$APPDATA\FreeScribe\settings.txt" 0 End
        ; Define MessageBox options with three buttons
        MessageBox MB_YESNOCANCEL|MB_ICONQUESTION|MB_DEFBUTTON3 "An old configuration file has been detected. Would you like to remove it to prevent conflict with new versions?" /SD IDCANCEL IDYES RemoveOldConfig IDNO KeepNetworkConfig
        ; IDCANCEL falls through to End (keep all)
        Goto End
        
        RemoveOldConfig:
            ClearErrors
            ; Remove all old configuration files
            RMDir /r "$APPDATA\FreeScribe"
            ${If} ${Errors}
                MessageBox MB_RETRYCANCEL "Unable to remove old configuration. Please close any applications using these files and try again." IDRETRY RemoveOldConfig IDCANCEL ConfigFilesFailed
            ${EndIf}
            Goto End
            
        KeepNetworkConfig:
            MessageBox MB_YESNO|MB_ICONQUESTION "Keep network config? This will preserve only network-related settings and remove other configuration files." IDYES KeepNetworkConfigOnly IDNO End
            
        KeepNetworkConfigOnly:
            ClearErrors
            ; Create a backup of the settings.txt file
            CopyFiles "$APPDATA\FreeScribe\settings.txt" "$APPDATA\FreeScribe\settings.txt.bak"
            
            ; Read the network configuration from settings.txt
            FileOpen $0 "$APPDATA\FreeScribe\settings.txt.bak" r
            ${If} ${Errors}
                MessageBox MB_OK|MB_ICONEXCLAMATION "Could not read settings file. Keeping all configuration files."
                Delete "$APPDATA\FreeScribe\settings.txt.bak"
                Goto End
            ${EndIf}
            
            ; Create a new settings file with only the requested configuration
            FileOpen $1 "$APPDATA\FreeScribe\settings.txt.new" w
            ${If} ${Errors}
                FileClose $0
                MessageBox MB_OK|MB_ICONEXCLAMATION "Could not create new settings file. Keeping all configuration files."
                Delete "$APPDATA\FreeScribe\settings.txt.bak"
                Goto End
            ${EndIf}
            
            ; Start with the opening bracket for JSON
            FileWrite $1 "{$\r$\n"
            FileWrite $1 '"openai_api_key": "None",$\r$\n'
            FileWrite $1 '"editable_settings": {$\r$\n'
            
            ; Variables to store extracted settings
            Var /GLOBAL AI_Server_Endpoint
            Var /GLOBAL AI_Self_Signed
            Var /GLOBAL Built_In_AI_Processing
            Var /GLOBAL Built_In_Speech2Text
            Var /GLOBAL S2T_Endpoint
            Var /GLOBAL S2T_API_Key
            Var /GLOBAL S2T_Self_Signed
            
            ; Default values in case not found
            StrCpy $AI_Server_Endpoint '"AI Server Endpoint": "https://localhost:3334/v1"'
            StrCpy $AI_Self_Signed '"AI Server Self-Signed Certificates": 0'
            StrCpy $Built_In_AI_Processing '"Built-in AI Processing": 1'
            StrCpy $Built_In_Speech2Text '"Built-in Speech2Text": 1'
            StrCpy $S2T_Endpoint '"Speech2Text (Whisper) Endpoint": "https://localhost:2224/whisperaudio"'
            StrCpy $S2T_API_Key '"Speech2Text (Whisper) API Key": ""'
            StrCpy $S2T_Self_Signed '"S2T Server Self-Signed Certificates": 0'
            
            ; Read settings file line by line to extract the values we want to keep
            Loop:
                FileRead $0 $2
                ${If} ${Errors}
                    Goto EndLoop
                ${EndIf}
                
                ; Check if line contains settings we want to keep
                ${If} $2 =~ '"AI Server Endpoint"'
                    StrCpy $AI_Server_Endpoint $2
                ${ElseIf} $2 =~ '"AI Server Self-Signed Certificates"'
                    StrCpy $AI_Self_Signed $2
                ${ElseIf} $2 =~ '"Built-in AI Processing"'
                    StrCpy $Built_In_AI_Processing $2
                ${ElseIf} $2 =~ '"Built-in Speech2Text"'
                    StrCpy $Built_In_Speech2Text $2
                ${ElseIf} $2 =~ '"Speech2Text \(Whisper\) Endpoint"'
                    StrCpy $S2T_Endpoint $2
                ${ElseIf} $2 =~ '"Speech2Text \(Whisper\) API Key"'
                    StrCpy $S2T_API_Key $2
                ${ElseIf} $2 =~ '"S2T Server Self-Signed Certificates"'
                    StrCpy $S2T_Self_Signed $2
                ${EndIf}
                
                Goto Loop
            EndLoop:
            
            ; Write the extracted settings to the new file
            ; Remove trailing commas if present in the extracted settings
            ${WordReplace} $AI_Server_Endpoint ",$" "" "+" $AI_Server_Endpoint
            ${WordReplace} $AI_Self_Signed ",$" "" "+" $AI_Self_Signed
            ${WordReplace} $Built_In_AI_Processing ",$" "" "+" $Built_In_AI_Processing
            ${WordReplace} $Built_In_Speech2Text ",$" "" "+" $Built_In_Speech2Text
            ${WordReplace} $S2T_Endpoint ",$" "" "+" $S2T_Endpoint
            ${WordReplace} $S2T_API_Key ",$" "" "+" $S2T_API_Key
            ${WordReplace} $S2T_Self_Signed ",$" "" "+" $S2T_Self_Signed
            
            ; Write each setting with comma
            FileWrite $1 "$AI_Server_Endpoint,$\r$\n"
            FileWrite $1 "$AI_Self_Signed,$\r$\n"
            FileWrite $1 "$Built_In_AI_Processing,$\r$\n"
            FileWrite $1 "$Built_In_Speech2Text,$\r$\n"
            FileWrite $1 "$S2T_Endpoint,$\r$\n"
            FileWrite $1 "$S2T_API_Key,$\r$\n"
            
            ; Last setting without comma
            FileWrite $1 "$S2T_Self_Signed$\r$\n"
            
            ; Close the JSON object
            FileWrite $1 "},$\r$\n"
            FileWrite $1 '"app_version": "aplha123"$\r$\n'
            FileWrite $1 "}"
            
            ; Close both files
            FileClose $0
            FileClose $1
            
            ; Remove all other configuration files except the new settings
            FindFirst $0 $1 "$APPDATA\FreeScribe\*.*"
            DeleteLoop:
                StrCmp $1 "" DeleteLoopEnd
                StrCmp $1 "." NextFile
                StrCmp $1 ".." NextFile
                StrCmp $1 "settings.txt.new" NextFile
                Delete "$APPDATA\FreeScribe\$1"
                NextFile:
                FindNext $0 $1
                Goto DeleteLoop
            DeleteLoopEnd:
            FindClose $0
            
            ; Remove directories
            RMDir /r "$APPDATA\FreeScribe\logs"
            RMDir /r "$APPDATA\FreeScribe\temp"
            RMDir /r "$APPDATA\FreeScribe\cache"
            
            ; Replace the old settings file with the new one
            Delete "$APPDATA\FreeScribe\settings.txt"
            Rename "$APPDATA\FreeScribe\settings.txt.new" "$APPDATA\FreeScribe\settings.txt"
            
            ; Clean up backup
            Delete "$APPDATA\FreeScribe\settings.txt.bak"
            
            MessageBox MB_OK "Network configuration preserved. All other settings were reset."
            Goto End
            
    ConfigFilesFailed:
        MessageBox MB_OK|MB_ICONEXCLAMATION "Old configuration files could not be fully processed. Proceeding with installation."
    End:
FunctionEnd

; Define the section of the installer
Section "MainSection" SEC01
    Call CleanUninstall
    Call CheckForOldConfig
    ; Set output path to the installation directory
    SetOutPath "$INSTDIR"

    ${If} $SELECTED_OPTION == "CPU"
        ; Add files to the installer
        File /r "..\dist\freescribe-client-cpu\freescribe-client-cpu.exe"
        Rename "$INSTDIR\freescribe-client-cpu.exe" "$INSTDIR\freescribe-client.exe"
        File /r "..\dist\freescribe-client-cpu\_internal"
    ${EndIf}

    ${If} $SELECTED_OPTION == "NVIDIA"
        ; Add files to the installer
        File /r "..\dist\freescribe-client-nvidia\freescribe-client-nvidia.exe"
        Rename "$INSTDIR\freescribe-client-nvidia.exe" "$INSTDIR\freescribe-client.exe"
        File /r "..\dist\freescribe-client-nvidia\_internal"
    ${EndIf}

    ; Install version file to both nvidia and cpu directories for version checking
    SetOutPath "$INSTDIR\_internal"
    File ".\__version__"

    SetOutPath "$INSTDIR"

    ; Create a start menu shortcut
    CreateDirectory "$SMPROGRAMS\FreeScribe"
    CreateShortcut "$SMPROGRAMS\FreeScribe\FreeScribe.lnk" "$INSTDIR\freescribe-client.exe"

    ; Create an uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "GGUF Installs" GGUF_INSTALLS
    AddSize 2800000 ; Add the size in kilobytes for the models

    CreateDirectory "$INSTDIR\models"
    SetOutPath "$INSTDIR\models"

    ; Copy the license
    File ".\assets\gemma_license.txt"

    ; Check if the file already exists
    IfFileExists "$INSTDIR\models\gemma-2-2b-it-Q8_0.gguf" 0 +2
    Goto SkipDownload

    ; Install the gemma 2 q8
    inetc::get /TIMEOUT=30000 "https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q8_0.gguf?download=true" "$INSTDIR\models\gemma-2-2b-it-Q8_0.gguf" /END

SkipDownload:
    SetOutPath "$INSTDIR"

SectionEnd

; Define the uninstaller section
Section "Uninstall"
    ; Remove the installation directory and all its contents
    RMDir /r "$INSTDIR"

    ; Remove the start menu shortcut
    Delete "$SMPROGRAMS\FreeScribe\FreeScribe.lnk"
    RMDir "$SMPROGRAMS\FreeScribe"

    ; Remove the uninstaller entry from the Control Panel
    Delete "$INSTDIR\Uninstall.exe"

    RemoveConfigFiles:
        ; Remove configuration files if the checkbox is selected
        ${If} $REMOVE_CONFIG == ${BST_CHECKED}
            ClearErrors
            RMDir /r "$APPDATA\FreeScribe"
            ${If} ${Errors}
                MessageBox MB_RETRYCANCEL "Unable to remove old configuration. Please close any applications using these files and try again." IDRETRY RemoveConfigFiles IDCANCEL ConfigFilesFailed
            ${EndIf}
        ${EndIf}

    ; Show message when uninstallation is complete
    MessageBox MB_OK "FreeScribe has been successfully uninstalled."
    Goto EndUninstall

    ConfigFilesFailed:
        MessageBox MB_OK|MB_ICONEXCLAMATION "FreeScribe has been successfully uninstalled, but the configuration files could not be removed. Please close any applications using these files and try again."
    EndUninstall:
SectionEnd

# Variables for checkboxes
Var DesktopShortcutCheckbox
Var StartMenuCheckbox
Var RunAppCheckbox

Function CustomizeFinishPage
    !insertmacro MUI_HEADER_TEXT "Installation Complete" "Please select your preferences and close the installer."

    nsDialogs::Create 1018
    Pop $0

    ${If} $0 == error
        Abort
    ${EndIf}

    # Run App Checkbox
    ${NSD_CreateCheckbox} 10u 10u 100% 12u "Run FreeScribe after installation"
    Pop $RunAppCheckbox
    ${NSD_SetState} $RunAppCheckbox ${BST_CHECKED}

    # Desktop Shortcut Checkbox
    ${NSD_CreateCheckbox} 10u 30u 100% 12u "Create Desktop Shortcut"
    Pop $DesktopShortcutCheckbox
    ${NSD_SetState} $DesktopShortcutCheckbox ${BST_CHECKED}

    # Start Menu Checkbox
    ${NSD_CreateCheckbox} 10u 50u 100% 12u "Add to Start Menu"
    Pop $StartMenuCheckbox
    ${NSD_SetState} $StartMenuCheckbox ${BST_CHECKED}

    nsDialogs::Show
FunctionEnd

Function RunApp
    ${NSD_GetState} $RunAppCheckbox $0
    ${If} $0 == ${BST_CHECKED}
        Exec '"$INSTDIR\freescribe-client.exe"'
    ${EndIf}

    # Check Desktop Shortcut
    ${NSD_GetState} $DesktopShortcutCheckbox $0
    StrCmp $0 ${BST_CHECKED} +2
        Goto SkipDesktopShortcut
    CreateShortcut "$DESKTOP\FreeScribe.lnk" "$INSTDIR\freescribe-client.exe"
    SkipDesktopShortcut:

    # Check Start Menu
    ${NSD_GetState} $StartMenuCheckbox $0
    StrCmp $0 ${BST_CHECKED} +2
        Goto SkipStartMenu
    CreateDirectory "$SMPROGRAMS\FreeScribe"
    CreateShortcut "$SMPROGRAMS\FreeScribe\FreeScribe.lnk" "$INSTDIR\freescribe-client.exe"
    SkipStartMenu:
FunctionEnd

; Function to execute when leaving the InstallFiles page
; Goes to the next page after the installation is complete
Function InsfilesPageLeave
    SetAutoClose true
FunctionEnd

Function CheckCudaAvailability
    nsExec::ExecToStack 'nvcc --version'
    Pop $0 ; Return value

    ${If} $0 != 0
        MessageBox MB_OK "CUDA is not available. Please ensure 'nvcc' is installed and added to the PATH and restart the installer. Download it from: https://developer.nvidia.com/cuda-downloads"
        Quit
    ${EndIf}
FunctionEnd

Function CheckNvidiaDrivers
    Var /GLOBAL DriverVersion

    ; Try to read from the registry
    SetRegView 64
    ReadRegStr $DriverVersion HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{B2FE1952-0186-46C3-BAEC-A80AA35AC5B8}_Display.Driver" "DisplayVersion"

    ${If} $DriverVersion == ""
        ; Fallback to 32-bit registry view
        SetRegView 32
        ReadRegStr $DriverVersion HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{B2FE1952-0186-46C3-BAEC-A80AA35AC5B8}_Display.Driver" "DisplayVersion"
    ${EndIf}

    ; No NVIDIA drivers detected - show error message
    ${If} $DriverVersion == ""
        MessageBox MB_OK "No valid NVIDIA device detected (Drivers Missing). This program relies on an NVIDIA GPU to run. Functionality is not guaranteed without an NVIDIA GPU."
        Goto driver_check_end
    ${EndIf}

    ; Push the version number to the stack
    Push $DriverVersion
    ; Push min driver version
    Push ${MIN_CUDA_DRIVER_VERSION}

    Call CompareVersions

    Pop $0 ; Get the return value

    ${If} $0 == 1
        MessageBox MB_OK "Your NVIDIA driver version ($DriverVersion) is older than the minimum required version (${MIN_CUDA_DRIVER_VERSION}). Please update at https://www.nvidia.com/en-us/drivers/. Then continue with the installation."
        Abort
    ${EndIf}

    ; Check for CUDA availability
    Call CheckCudaAvailability

    driver_check_end:
FunctionEnd

;------------------------------------------------------------------------------
; Function: CompareVersions
; Purpose: Compares two version numbers in format "X.Y" (e.g., "1.0", "2.3")
;
; Parameters:
;   Stack 1 (bottom): First version string to compare
;   Stack 0 (top): Second version string to compare
;
; Returns:
;   0: Versions are equal
;   1: First version is less than second version
;   2: First version is greater than second version
;
; Example:
;   Push "1.0"    ; First version
;   Push "2.0"    ; Second version
;   Call CompareVersions
;   Pop $R0       ; $R0 will contain 1 (1.0 < 2.0)
;------------------------------------------------------------------------------
Function CompareVersions
    Exch $R0      ; Get second version from stack into $R0
    Exch
    Exch $R1      ; Get first version from stack into $R1
    Push $R2
    Push $R3
    Push $R4
    Push $R5

    ; Split version strings into major and minor numbers
    ${WordFind} $R1 "." "+1" $R2    ; Extract major number from first version
    ${WordFind} $R1 "." "+2" $R3    ; Extract minor number from first version
    ${WordFind} $R0 "." "+1" $R4    ; Extract major number from second version
    ${WordFind} $R0 "." "+2" $R5    ; Extract minor number from second version

    ; Convert to comparable numbers:
    ; Multiply major version by 1000 to handle minor version properly
    IntOp $R2 $R2 * 1000            ; Convert first version major number
    IntOp $R4 $R4 * 1000            ; Convert second version major number

    ; Add minor numbers to create complete comparable values
    IntOp $R2 $R2 + $R3             ; First version complete number
    IntOp $R4 $R4 + $R5             ; Second version complete number

    ; Compare versions and set return value
    ${If} $R2 < $R4                 ; If first version is less than second
        StrCpy $R0 1
    ${ElseIf} $R2 > $R4             ; If first version is greater than second
        StrCpy $R0 2
    ${Else}                         ; If versions are equal
        StrCpy $R0 0
    ${EndIf}

    ; Restore registers from stack
    Pop $R5
    Pop $R4
    Pop $R3
    Pop $R2
    Pop $R1
    Exch $R0                        ; Put return value on stack
FunctionEnd

Function un.CreateRemoveConfigFilesPage
    !insertmacro MUI_HEADER_TEXT "Remove Configuration Files" "Do you want to remove the configuration files (e.g., settings)?"

    nsDialogs::Create 1018
    Pop $0

    ${If} $0 == error
        Abort
    ${EndIf}

    ${NSD_CreateCheckbox} 0 20u 100% 12u "Remove configuration files"
    Pop $REMOVE_CONFIG_CHECKBOX
    ${NSD_SetState} $REMOVE_CONFIG_CHECKBOX ${BST_CHECKED}

    nsDialogs::Show
FunctionEnd

Function un.RemoveConfigFilesPageLeave
    ${NSD_GetState} $REMOVE_CONFIG_CHECKBOX $REMOVE_CONFIG
FunctionEnd

; Define the uninstaller pages first
UninstPage custom un.CreateRunningInstancePage
!insertmacro MUI_UNPAGE_CONFIRM
UninstPage custom un.CreateRemoveConfigFilesPage un.RemoveConfigFilesPageLeave
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Define installer pages
Page custom CreateRunningInstancePage
!insertmacro MUI_PAGE_LICENSE ".\assets\License.txt"
Page Custom ARCHITECTURE_SELECT ARCHITECTURE_SELECT_LEAVE
!insertmacro MUI_PAGE_DIRECTORY
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE InsfilesPageLeave
!insertmacro MUI_PAGE_INSTFILES
Page Custom CustomizeFinishPage RunApp

; Define the languages
!insertmacro MUI_LANGUAGE English
