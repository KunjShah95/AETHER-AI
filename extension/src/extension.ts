import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { AssistantPanel, QuickCommand } from './assistantPanel';

const MAX_LOG_LINES = 250;
const QUICK_COMMANDS: QuickCommand[] = [
  { label: 'Show help', description: 'Print the full command catalog', command: '/help' },
  { label: 'Active model status', description: 'Show configured AI models and health', command: '/models' },
  { label: 'Current model', description: 'Describe the model that is running now', command: '/current-model' },
  { label: 'Switch to GPT-5.1', description: 'Enable GPT-5.1-Codex-Mini (Preview) everywhere', command: '/switch gpt-5.1-codex-mini' },
  { label: 'System status', description: 'Detailed health of the assistant', command: '/status' },
  { label: 'Git status', description: 'Show repository staging info', command: '/git status' },
  { label: 'List plugins', description: 'See registered plugin commands', command: '/plugins list' },
  { label: 'Tasks summary', description: 'List open tasks', command: '/tasks' },
  { label: 'View TODOs', description: 'Summarize TODO and FIXME entries', command: '/todos' },
  { label: 'Quick note', description: 'Capture a note now', command: '/note extension quick note' }
];

let assistantProcess: cp.ChildProcessWithoutNullStreams | undefined;
let outputChannel: vscode.OutputChannel;
let logLines: string[] = [];

export function activate(context: vscode.ExtensionContext) {
  outputChannel = vscode.window.createOutputChannel('Aether AI Terminal');
  context.subscriptions.push(outputChannel);

  registerCommand(context, 'aetherAi.startAssistant', startAssistant);
  registerCommand(context, 'aetherAi.stopAssistant', stopAssistant);
  registerCommand(context, 'aetherAi.sendQuickCommand', sendQuickCommand);
  registerCommand(context, 'aetherAi.openAssistantPanel', () => openAssistantPanel(context));
  registerCommand(context, 'aetherAi.insertTimestamp', insertTimestamp);

  const autoStart = getConfig('autoStart', false);
  if (autoStart) {
    void startAssistant();
  }

  context.subscriptions.push({
    dispose: () => {
      assistantProcess?.kill();
      assistantProcess = undefined;
    }
  });
}

function registerCommand(
  context: vscode.ExtensionContext,
  command: string,
  callback: (...args: any[]) => unknown
) {
  context.subscriptions.push(vscode.commands.registerCommand(command, callback));
}

function getConfig<T>(key: string, defaultValue: T): T {
  return vscode.workspace.getConfiguration('aetherAiAssistant').get<T>(key, defaultValue) || defaultValue;
}

async function startAssistant() {
  if (assistantProcess) {
    vscode.window.showInformationMessage('The Aether AI assistant is already running.');
    return;
  }

  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    vscode.window.showErrorMessage('Open the workspace that contains the Aether AI repository.');
    return;
  }

  const workspaceRoot = folders[0].uri.fsPath;
  const assistantScript = path.join(workspaceRoot, 'terminal', 'main.py');
  if (!fs.existsSync(assistantScript)) {
    vscode.window.showErrorMessage('terminal/main.py cannot be found in the workspace root.');
    return;
  }

  const pythonPath = getConfig('pythonPath', 'python');
  logLines = [];
  outputChannel.clear();

  try {
    assistantProcess = cp.spawn(
      pythonPath,
      ['-u', assistantScript, '--cli'],
      {
        cwd: workspaceRoot,
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
      }
    );
  } catch (err) {
    vscode.window.showErrorMessage(`Failed to start assistant: ${err}`);
    assistantProcess = undefined;
    return;
  }

  const preferredModel = getConfig('preferredModel', 'gpt-5.1-codex-mini');
  recordLog(`Preferred model: ${preferredModel}`);

  recordLog('Assistant starting…');
  updateStatus(`Running via ${pythonPath}`);

  assistantProcess.stdout.on('data', (chunk) => {
    recordLog(chunk.toString().trim());
  });

  assistantProcess.stderr.on('data', (chunk) => {
    recordLog(chunk.toString().trim());
  });

  assistantProcess.on('error', (error) => {
    recordLog(`Process error: ${error.message}`);
    assistantProcess = undefined;
    updateStatus('Stopped (error)');
  });

  assistantProcess.on('exit', (code, signal) => {
    recordLog(`Assistant exited${signal ? ` (signal ${signal})` : ''}${code !== null ? ` with code ${code}` : ''}`);
    assistantProcess = undefined;
    updateStatus('Stopped');
  });
}

async function stopAssistant() {
  if (!assistantProcess) {
    vscode.window.showInformationMessage('The assistant is not running.');
    return;
  }

  recordLog('Stopping assistant…');
  assistantProcess.kill('SIGINT');
  assistantProcess = undefined;
  updateStatus('Stopped');
}

async function sendQuickCommand() {
  if (!assistantProcess) {
    vscode.window.showWarningMessage('Start the assistant first.');
    return;
  }

  const item = await vscode.window.showQuickPick(
    QUICK_COMMANDS.map((command) => ({
      label: command.label,
      description: command.description,
      detail: command.command
    })),
    {
      placeHolder: 'Choose a quick Aether AI command'
    }
  );

  if (item?.detail) {
    sendToAssistant(item.detail);
  }
}

function sendToAssistant(command: string) {
  if (!assistantProcess) {
    recordLog('Command ignored — assistant is not running.');
    return;
  }

  recordLog(`> ${command}`);
  try {
    assistantProcess.stdin.write(`${normalizeCommand(command)}\n`);
  } catch (error) {
    recordLog(`Failed to send command: ${error instanceof Error ? error.message : error}`);
  }
}

function normalizeCommand(command: string): string {
  if (command === '/switch gpt-5.1-codex-mini') {
    recordLog('Preview mode: routing GPT-5.1-Codex-Mini requests through the ChatGPT backend.');
    return '/switch chatgpt';
  }
  return command;
}

function openAssistantPanel(context: vscode.ExtensionContext) {
  AssistantPanel.createOrShow(
    context.extensionUri,
    QUICK_COMMANDS,
    logLines,
    assistantProcess ? 'Running' : 'Stopped',
    sendToAssistant
  );
}

function recordLog(entry: string) {
  const trimmed = entry.trim();
  if (!trimmed) {
    return;
  }

  logLines.push(trimmed);
  if (logLines.length > MAX_LOG_LINES) {
    logLines = logLines.slice(-MAX_LOG_LINES);
  }

  outputChannel.appendLine(trimmed);
  updatePanelLog();
}

function updatePanelLog() {
  AssistantPanel.currentPanel?.postLog(logLines);
}

function updateStatus(status: string) {
  AssistantPanel.currentPanel?.postStatus(status);
}

function insertTimestamp() {
  const editor = vscode.window.activeTextEditor;
  if (editor) {
    const timestamp = new Date().toISOString();
    editor.edit(editBuilder => {
      editBuilder.insert(editor.selection.active, timestamp);
    });
  }
}

export function deactivate() {
  assistantProcess?.kill();
  assistantProcess = undefined;
}
