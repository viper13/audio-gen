Option Explicit
Dim fso, rootDir, shell, args, extra, i, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
rootDir = fso.GetParentFolderName(WScript.ScriptFullName)
extra = ""
For i = 0 To WScript.Arguments.Count - 1
    extra = extra & " " & WScript.Arguments(i)
Next
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & rootDir & "\service-runner.ps1""" & extra
Set shell = CreateObject("WScript.Shell")
' WindowStyle 0 = completely hidden, WaitOnReturn = True so Task Scheduler tracks running status
WScript.Quit shell.Run(cmd, 0, True)
