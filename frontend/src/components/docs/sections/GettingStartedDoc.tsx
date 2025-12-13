import { motion } from 'framer-motion';
import { Download, Code, Key, Play, CheckCircle, AlertCircle } from 'lucide-react';
import CodeBlock from '../CodeBlock';

const GettingStartedDoc = () => {
    const steps = [
        {
            icon: Download,
            title: 'Installation',
            color: 'from-purple-500 to-blue-500'
        },
        {
            icon: Key,
            title: 'Configuration',
            color: 'from-blue-500 to-cyan-500'
        },
        {
            icon: Play,
            title: 'Quick Start',
            color: 'from-cyan-500 to-green-500'
        },
    ];

    return (
        <div className="space-y-10">
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
            >
                <h1 className="text-4xl lg:text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                    Getting Started
                </h1>
                <p className="text-lg text-slate-300">
                    Get AETHER AI up and running in minutes with this comprehensive guide.
                </p>
            </motion.div>

            {/* Progress Steps */}
            <div className="grid md:grid-cols-3 gap-4">
                {steps.map((step, index) => {
                    const Icon = step.icon;
                    return (
                        <motion.div
                            key={step.title}
                            initial={{ y: 20, opacity: 0 }}
                            animate={{ y: 0, opacity: 1 }}
                            transition={{ delay: index * 0.1 }}
                            className="relative"
                        >
                            <div className="flex items-center gap-3 bg-slate-800/30 border border-purple-500/20 rounded-xl p-4">
                                <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${step.color} flex items-center justify-center shrink-0`}>
                                    <Icon className="w-5 h-5 text-white" />
                                </div>
                                <div>
                                    <p className="text-xs text-slate-400">Step {index + 1}</p>
                                    <p className="font-semibold text-white">{step.title}</p>
                                </div>
                            </div>
                        </motion.div>
                    );
                })}
            </div>

            {/* Prerequisites */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-500/30 rounded-xl p-6"
            >
                <div className="flex items-start gap-3">
                    <AlertCircle className="w-6 h-6 text-blue-400 shrink-0 mt-1" />
                    <div>
                        <h3 className="text-lg font-semibold text-white mb-2">Prerequisites</h3>
                        <ul className="space-y-2 text-slate-300">
                            <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-400" />
                                <span><strong>Python 3.9+</strong> installed on your system</span>
                            </li>
                            <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-400" />
                                <span>Internet access for cloud models (Gemini/Groq/HF/ChatGPT/MCP)</span>
                            </li>
                            <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-yellow-400" />
                                <span><strong>Optional:</strong> <a href="https://ollama.ai" className="text-purple-400 hover:text-purple-300 underline">Ollama</a> for local models</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </motion.div>

            {/* Installation Options */}
            <div className="space-y-6">
                <h2 className="text-2xl font-bold text-white">Installation Options</h2>

                {/* Option A */}
                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.4 }}
                    className="space-y-4"
                >
                    <div className="flex items-center gap-3">
                        <div className="px-3 py-1 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg text-sm font-semibold">
                            RECOMMENDED
                        </div>
                        <h3 className="text-xl font-semibold text-white">Option A: One-Click OS Scripts</h3>
                    </div>

                    <p className="text-slate-300">
                        We provide a single canonical installer per platform under <code className="px-2 py-1 bg-slate-800 rounded text-purple-400">dist/</code>
                    </p>

                    <div className="grid md:grid-cols-3 gap-4">
                        <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                            <p className="text-sm text-slate-400 mb-2">🪟 Windows</p>
                            <code className="text-xs text-purple-400 break-all">dist/install_windows.bat</code>
                        </div>
                        <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                            <p className="text-sm text-slate-400 mb-2">🍎 macOS</p>
                            <code className="text-xs text-purple-400 break-all">dist/install_mac.sh</code>
                        </div>
                        <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                            <p className="text-sm text-slate-400 mb-2">🐧 Linux</p>
                            <code className="text-xs text-purple-400 break-all">dist/install_linux.sh</code>
                        </div>
                    </div>

                    <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
                        <p className="text-sm text-yellow-300">
                            <strong>Security Note:</strong> Before running an installer, verify the SHA256 checksum in <code className="px-2 py-1 bg-slate-800 rounded">dist/SHA256SUMS.txt</code>
                        </p>
                    </div>
                </motion.div>

                {/* Option B */}
                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.5 }}
                    className="space-y-4"
                >
                    <h3 className="text-xl font-semibold text-white">Option B: From Source (Development)</h3>

                    <CodeBlock language="bash" code={`# Clone the repository
git clone https://github.com/KunjShah95/NEXUS-AI.io.git
cd NEXUS-AI.io

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env   # Add your API keys

# Run the application
python terminal/main.py`} />
                </motion.div>
            </div>

            {/* Configuration */}
            <div className="space-y-4">
                <h2 className="text-2xl font-bold text-white">Configuration</h2>

                <p className="text-slate-300">
                    Create and edit a <code className="px-2 py-1 bg-slate-800 rounded text-purple-400">.env</code> file at the project root:
                </p>

                <CodeBlock language="env" code={`GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here
MCP_API_KEY=your_mcp_api_key_here`} />

                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                    <p className="text-sm text-blue-300">
                        <strong>💡 Note on Ollama:</strong> Ollama does not require an API key but must be installed and the daemon running.
                    </p>
                </div>

                <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                    <h4 className="font-semibold text-white mb-2">Config File Locations</h4>
                    <ul className="space-y-2 text-slate-300 text-sm">
                        <li><strong>Windows:</strong> <code className="px-2 py-1 bg-slate-800 rounded text-purple-400">%USERPROFILE%\.nexus</code></li>
                        <li><strong>macOS/Linux:</strong> <code className="px-2 py-1 bg-slate-800 rounded text-purple-400">~/.nexus</code></li>
                    </ul>
                </div>
            </div>

            {/* Quick Start */}
            <div className="space-y-4">
                <h2 className="text-2xl font-bold text-white">Quick Start</h2>

                <p className="text-slate-300">Launch the terminal assistant:</p>

                <CodeBlock language="bash" code={`python terminal/main.py`} />

                <div className="grid md:grid-cols-2 gap-4">
                    <div className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-lg p-4">
                        <h4 className="font-semibold text-white mb-2 flex items-center gap-2">
                            <Play className="w-5 h-5" />
                            Essential Commands
                        </h4>
                        <ul className="space-y-1 text-sm text-slate-300">
                            <li><code className="text-purple-400">/help</code> - View all commands</li>
                            <li><code className="text-purple-400">/switch [model]</code> - Change AI model</li>
                            <li><code className="text-purple-400">/ollama-models</code> - List local models</li>
                        </ul>
                    </div>

                    <div className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-lg p-4">
                        <h4 className="font-semibold text-white mb-2 flex items-center gap-2">
                            <CheckCircle className="w-5 h-5" />
                            You're Ready!
                        </h4>
                        <p className="text-sm text-slate-300">
                            Start chatting with your AI assistant. Try asking questions, running commands, or exploring features!
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default GettingStartedDoc;
