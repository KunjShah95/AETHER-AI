import { motion } from 'framer-motion';
import { Code, GitBranch, BookOpen, Rocket, TestTube, FolderTree } from 'lucide-react';
import CodeBlock from '../CodeBlock';

const DeveloperDoc = () => {
    const steps = [
        {
            icon: GitBranch,
            title: 'Clone Repository',
            color: 'from-purple-500 to-blue-500'
        },
        {
            icon: Code,
            title: 'Install Dependencies',
            color: 'from-blue-500 to-cyan-500'
        },
        {
            icon: Rocket,
            title: 'Run Application',
            color: 'from-cyan-500 to-green-500'
        },
    ];

    const projectStructure = [
        { path: '/', name: 'Project Root' },
        { path: 'terminal/', name: 'Main CLI application', highlight: true },
        { path: 'terminal/main.py', name: 'Entry point' },
        { path: 'terminal/requirements.txt', name: 'Python dependencies' },
        { path: 'frontend/', name: 'React website', highlight: true },
        { path: 'documentation-site/', name: 'Documentation' },
        { path: 'extension/', name: 'VSCode extension' },
        { path: '.env', name: 'Environment config' },
    ];

    return (
        <div className="space-y-10">
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
            >
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
                        <Code className="w-8 h-8 text-white" />
                    </div>
                    <div>
                        <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                            Developer Guide
                        </h1>
                        <p className="text-slate-400">Contribute to AETHER AI</p>
                    </div>
                </div>
                <p className="text-lg text-slate-300">
                    We welcome contributions to AETHER AI! This guide will help you set up your development environment and understand the project structure.
                </p>
            </motion.div>

            {/* Quick Start */}
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
                <h3 className="text-lg font-semibold text-white mb-4">Prerequisites</h3>
                <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                        <div className="flex items-center gap-2">
                            <span className="text-2xl">🐍</span>
                            <div>
                                <p className="font-semibold text-white">Python 3.9+</p>
                                <p className="text-sm text-slate-400">Required for terminal app</p>
                            </div>
                        </div>
                    </div>
                    <div className="space-y-2">
                        <div className="flex items-center gap-2">
                            <span className="text-2xl">📦</span>
                            <div>
                                <p className="font-semibold text-white">Git</p>
                                <p className="text-sm text-slate-400">Version control</p>
                            </div>
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* Installation Steps */}
            <div className="space-y-6">
                <h2 className="text-2xl font-bold text-white">Development Setup</h2>

                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.4 }}
                    className="space-y-4"
                >
                    <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                        <span className="flex items-center justify-center w-6 h-6 rounded-full bg-purple-500 text-white text-sm font-bold">1</span>
                        Clone the Repository
                    </h3>

                    <CodeBlock language="bash" code={`git clone https://github.com/KunjShah95/NEXUS-AI.io.git
cd NEXUS-AI.io`} />
                </motion.div>

                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.5 }}
                    className="space-y-4"
                >
                    <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                        <span className="flex items-center justify-center w-6 h-6 rounded-full bg-purple-500 text-white text-sm font-bold">2</span>
                        Create Virtual Environment (Recommended)
                    </h3>

                    <CodeBlock language="bash" code={`# Create virtual environment
python -m venv venv

# Activate on Windows
venv\\Scripts\\activate

# Activate on macOS/Linux
source venv/bin/activate`} />
                </motion.div>

                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.6 }}
                    className="space-y-4"
                >
                    <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                        <span className="flex items-center justify-center w-6 h-6 rounded-full bg-purple-500 text-white text-sm font-bold">3</span>
                        Install Dependencies
                    </h3>

                    <CodeBlock language="bash" code={`pip install -r requirements.txt`} />
                </motion.div>

                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.7 }}
                    className="space-y-4"
                >
                    <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                        <span className="flex items-center justify-center w-6 h-6 rounded-full bg-purple-500 text-white text-sm font-bold">4</span>
                        Configure Environment
                    </h3>

                    <p className="text-slate-300">
                        Copy <code className="px-2 py-1 bg-slate-800 rounded text-purple-400">.env.example</code> to <code className="px-2 py-1 bg-slate-800 rounded text-purple-400">.env</code> and add your API keys:
                    </p>

                    <CodeBlock language="bash" code={`cp .env.example .env
# Then edit .env with your favorite editor`} />
                </motion.div>

                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.8 }}
                    className="space-y-4"
                >
                    <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                        <span className="flex items-center justify-center w-6 h-6 rounded-full bg-purple-500 text-white text-sm font-bold">5</span>
                        Run the Application
                    </h3>

                    <CodeBlock language="bash" code={`python terminal/main.py`} />
                </motion.div>
            </div>

            {/* Project Structure */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.9 }}
                className="space-y-4"
            >
                <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                    <FolderTree className="w-6 h-6 text-purple-400" />
                    Project Structure
                </h2>

                <div className="bg-slate-900/50 border border-purple-500/20 rounded-xl overflow-hidden">
                    <div className="p-4 bg-slate-800/50 border-b border-slate-700/50">
                        <p className="text-sm text-slate-400">Directory Layout</p>
                    </div>
                    <div className="p-6 space-y-2 font-mono text-sm">
                        {projectStructure.map((item, index) => (
                            <motion.div
                                key={item.path}
                                initial={{ x: -20, opacity: 0 }}
                                animate={{ x: 0, opacity: 1 }}
                                transition={{ delay: index * 0.05 + 1.0 }}
                                className={`flex items-center gap-3 py-2 px-3 rounded-lg ${item.highlight
                                    ? 'bg-purple-500/10 border border-purple-500/30'
                                    : 'hover:bg-slate-800/50'
                                    }`}
                            >
                                <code className="text-purple-400">{item.path}</code>
                                <span className="text-slate-500">―</span>
                                <span className="text-slate-300">{item.name}</span>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </motion.div>

            {/* Contributing Workflow */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 1.1 }}
                className="space-y-4"
            >
                <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                    <GitBranch className="w-6 h-6 text-purple-400" />
                    Contributing Workflow
                </h2>

                <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                        <h3 className="font-semibold text-white">Standard Flow</h3>
                        <ol className="space-y-2 text-sm text-slate-300">
                            <li className="flex gap-2">
                                <span className="text-purple-400 font-bold">1.</span>
                                <span>Fork the repository on GitHub</span>
                            </li>
                            <li className="flex gap-2">
                                <span className="text-purple-400 font-bold">2.</span>
                                <span>Create a feature branch</span>
                            </li>
                            <li className="flex gap-2">
                                <span className="text-purple-400 font-bold">3.</span>
                                <span>Make your changes with clear commits</span>
                            </li>
                            <li className="flex gap-2">
                                <span className="text-purple-400 font-bold">4.</span>
                                <span>Push to your fork</span>
                            </li>
                            <li className="flex gap-2">
                                <span className="text-purple-400 font-bold">5.</span>
                                <span>Open a Pull Request</span>
                            </li>
                        </ol>
                    </div>

                    <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                        <h4 className="font-semibold text-white mb-3 text-sm">Example Commands</h4>
                        <CodeBlock language="bash" code={`# Create feature branch
git checkout -b feat/your-feature

# Make commits
git commit -m "feat: add new feature"

# Push to fork
git push origin feat/your-feature`} />
                    </div>
                </div>
            </motion.div>

            {/* Code Style */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 1.2 }}
                className="space-y-4"
            >
                <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                    <BookOpen className="w-6 h-6 text-purple-400" />
                    Code Style Guidelines
                </h2>

                <div className="grid md:grid-cols-3 gap-4">
                    <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                        <h4 className="font-semibold text-purple-400 mb-2">Python</h4>
                        <ul className="space-y-1 text-sm text-slate-300">
                            <li>• Follow PEP 8 guidelines</li>
                            <li>• Use type hints where possible</li>
                            <li>• Add docstrings to functions</li>
                            <li>• Keep functions focused</li>
                        </ul>
                    </div>

                    <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                        <h4 className="font-semibold text-blue-400 mb-2">Commits</h4>
                        <ul className="space-y-1 text-sm text-slate-300">
                            <li>• Use conventional commits</li>
                            <li>• Write clear messages</li>
                            <li>• Reference issues</li>
                            <li>• Keep commits atomic</li>
                        </ul>
                    </div>

                    <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                        <h4 className="font-semibold text-green-400 mb-2">Testing</h4>
                        <ul className="space-y-1 text-sm text-slate-300">
                            <li>• Add tests for new features</li>
                            <li>• Ensure tests pass</li>
                            <li>• Use pytest framework</li>
                            <li>• Aim for good coverage</li>
                        </ul>
                    </div>
                </div>
            </motion.div>

            {/* Testing */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 1.3 }}
                className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-xl p-6"
            >
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                    <TestTube className="w-5 h-5 text-green-400" />
                    Running Tests
                </h3>
                <p className="text-slate-300 mb-4">
                    Run unit tests to ensure your changes don't break existing functionality:
                </p>
                <CodeBlock language="bash" code={`python -m pytest terminal/tests/`} />
            </motion.div>

            {/* Get Help */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 1.4 }}
                className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-xl p-6"
            >
                <h3 className="text-lg font-semibold text-white mb-3">Need Help?</h3>
                <p className="text-slate-300 mb-4">We're here to help you contribute successfully!</p>
                <div className="grid md:grid-cols-3 gap-4">
                    <a href="https://github.com/KunjShah95/NEXUS-AI.io/issues"
                        className="flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors">
                        <span>→</span>
                        <span>GitHub Issues</span>
                    </a>
                    <a href="https://github.com/KunjShah95/NEXUS-AI.io/discussions"
                        className="flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors">
                        <span>→</span>
                        <span>Discussions</span>
                    </a>
                    <a href="https://github.com/KunjShah95/NEXUS-AI.io"
                        className="flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors">
                        <span>→</span>
                        <span>Main Repository</span>
                    </a>
                </div>
            </motion.div>
        </div>
    );
};

export default DeveloperDoc;
