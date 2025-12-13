"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const cp = __importStar(require("child_process"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const assistantPanel_1 = require("./assistantPanel");
const MAX_LOG_LINES = 250;
const QUICK_COMMANDS = [
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
let assistantProcess;
let outputChannel;
let logLines = [];
function activate(context) {
    outputChannel = vscode.window.createOutputChannel('Nexus AI Terminal');
    context.subscriptions.push(outputChannel);
    registerCommand(context, 'nexusAi.startAssistant', startAssistant);
    registerCommand(context, 'nexusAi.stopAssistant', stopAssistant);
    registerCommand(context, 'nexusAi.sendQuickCommand', sendQuickCommand);
    registerCommand(context, 'nexusAi.openAssistantPanel', () => openAssistantPanel(context));
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
function registerCommand(context, command, callback) {
    context.subscriptions.push(vscode.commands.registerCommand(command, callback));
}
function getConfig(key, defaultValue) {
    return vscode.workspace.getConfiguration('nexusAiAssistant').get(key, defaultValue) || defaultValue;
}
async function startAssistant() {
    if (assistantProcess) {
        vscode.window.showInformationMessage('The Nexus AI assistant is already running.');
        return;
    }
    const folders = vscode.workspace.workspaceFolders;
    if (!folders || folders.length === 0) {
        vscode.window.showErrorMessage('Open the workspace that contains the Nexus AI repository.');
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
        assistantProcess = cp.spawn(pythonPath, ['-u', assistantScript, '--cli'], {
            cwd: workspaceRoot,
            env: { ...process.env, PYTHONUNBUFFERED: '1' }
        });
    }
    catch (err) {
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
    const item = await vscode.window.showQuickPick(QUICK_COMMANDS.map((command) => ({
        label: command.label,
        description: command.description,
        detail: command.command
    })), {
        placeHolder: 'Choose a quick Nexus AI command'
    });
    if (item?.detail) {
        sendToAssistant(item.detail);
    }
}
function sendToAssistant(command) {
    if (!assistantProcess) {
        recordLog('Command ignored — assistant is not running.');
        return;
    }
    recordLog(`> ${command}`);
    try {
        assistantProcess.stdin.write(`${normalizeCommand(command)}\n`);
    }
    catch (error) {
        recordLog(`Failed to send command: ${error instanceof Error ? error.message : error}`);
    }
}
function normalizeCommand(command) {
    if (command === '/switch gpt-5.1-codex-mini') {
        recordLog('Preview mode: routing GPT-5.1-Codex-Mini requests through the ChatGPT backend.');
        return '/switch chatgpt';
    }
    return command;
}
function openAssistantPanel(context) {
    assistantPanel_1.AssistantPanel.createOrShow(context.extensionUri, QUICK_COMMANDS, logLines, assistantProcess ? 'Running' : 'Stopped', sendToAssistant);
}
function recordLog(entry) {
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
    assistantPanel_1.AssistantPanel.currentPanel?.postLog(logLines);
}
function updateStatus(status) {
    assistantPanel_1.AssistantPanel.currentPanel?.postStatus(status);
}
function deactivate() {
    assistantProcess?.kill();
    assistantProcess = undefined;
}
//# sourceMappingURL=extension.js.map