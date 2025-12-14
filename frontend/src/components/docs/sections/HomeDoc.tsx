import { motion } from 'framer-motion';
import { Terminal, Zap, Shield, Code, CheckSquare, Globe, Cpu } from 'lucide-react';

const HomeDoc = () => {
    const features = [
        {
            icon: Zap,
            title: 'Multi-Model Switching',
            description: 'Switch between Gemini, Groq, Ollama, HuggingFace, ChatGPT, and MCP',
            color: 'from-yellow-500 to-orange-500'
        },
        {
            icon: Cpu,
            title: 'Local AI via Ollama',
            description: 'Run models locally with complete privacy and offline access',
            color: 'from-blue-500 to-cyan-500'
        },
        {
            icon: Shield,
            title: 'Secure by Default',
            description: 'Input sanitization, safe command allowlist, and boundary checks',
            color: 'from-green-500 to-emerald-500'
        },
        {
            icon: Globe,
            title: 'Autonomous Agents',
            description: 'Multi-agent system with efficient Web Agent and specialized Personas',
            color: 'from-purple-500 to-pink-500'
        },
        {
            icon: Code,
            title: 'Developer Tools',
            description: 'Code review, refactoring, TODO extraction, and Git helpers',
            color: 'from-red-500 to-rose-500'
        },
        {
            icon: CheckSquare,
            title: 'Mission Control',
            description: 'AutoPilot workflows, TUI Dashboard, and real-time monitoring',
            color: 'from-indigo-500 to-violet-500'
        },
    ];

    return (
        <div className="space-y-12">
            {/* Hero Section */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                className="text-center space-y-6"
            >
                <div className="inline-flex items-center gap-3 bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/30 rounded-full px-6 py-2">
                    <Terminal className="w-5 h-5 text-purple-400" />
                    <span className="text-sm font-medium bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                        Version 4.0.0
                    </span>
                </div>

                <h1 className="text-5xl lg:text-6xl font-bold bg-gradient-to-r from-purple-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                    AETHER AI Terminal Assistant
                </h1>

                <p className="text-xl text-slate-300 max-w-3xl mx-auto leading-relaxed">
                    Production‑ready, secure, multi‑model AI for your terminal. Switch between multiple AI providers
                    with one CLI, enriched by security controls, utilities, and an extensible modular architecture.
                </p>

                <div className="flex gap-4 justify-center flex-wrap">
                    <span className="px-4 py-2 bg-slate-800/50 border border-purple-500/20 rounded-lg text-sm">
                        ⚡ Multi-Model
                    </span>
                    <span className="px-4 py-2 bg-slate-800/50 border border-blue-500/20 rounded-lg text-sm">
                        🔒 Secure
                    </span>
                    <span className="px-4 py-2 bg-slate-800/50 border border-green-500/20 rounded-lg text-sm">
                        🚀 Fast
                    </span>
                    <span className="px-4 py-2 bg-slate-800/50 border border-pink-500/20 rounded-lg text-sm">
                        🎨 Customizable
                    </span>
                </div>
            </motion.div>

            {/* Features Grid */}
            <div>
                <h2 className="text-3xl font-bold mb-8 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                    Key Features
                </h2>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {features.map((feature, index) => {
                        const Icon = feature.icon;
                        return (
                            <motion.div
                                key={feature.title}
                                initial={{ y: 20, opacity: 0 }}
                                animate={{ y: 0, opacity: 1 }}
                                transition={{ delay: index * 0.1 }}
                                className="group relative overflow-hidden bg-slate-800/30 backdrop-blur-sm border border-purple-500/20 rounded-xl p-6 hover:border-purple-500/40 transition-all duration-300"
                            >
                                {/* Hover effect */}
                                <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />

                                <div className="relative space-y-3">
                                    <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center`}>
                                        <Icon className="w-6 h-6 text-white" />
                                    </div>

                                    <h3 className="text-lg font-semibold text-white">
                                        {feature.title}
                                    </h3>

                                    <p className="text-slate-400 text-sm leading-relaxed">
                                        {feature.description}
                                    </p>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Quick Links */}
            <div className="grid md:grid-cols-3 gap-6">
                <motion.a
                    href="#getting-started"
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.6 }}
                    className="group relative overflow-hidden bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-500/30 rounded-xl p-6 hover:border-purple-500/50 transition-all duration-300"
                >
                    <div className="relative space-y-2">
                        <h3 className="text-lg font-semibold text-white group-hover:text-purple-400 transition-colors">
                            Getting Started →
                        </h3>
                        <p className="text-slate-400 text-sm">
                            Quick installation and setup guide
                        </p>
                    </div>
                </motion.a>

                <motion.a
                    href="#commands"
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.7 }}
                    className="group relative overflow-hidden bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border border-blue-500/30 rounded-xl p-6 hover:border-blue-500/50 transition-all duration-300"
                >
                    <div className="relative space-y-2">
                        <h3 className="text-lg font-semibold text-white group-hover:text-blue-400 transition-colors">
                            Commands Reference →
                        </h3>
                        <p className="text-slate-400 text-sm">
                            Complete command documentation
                        </p>
                    </div>
                </motion.a>

                <motion.a
                    href="#developer"
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.8 }}
                    className="group relative overflow-hidden bg-gradient-to-br from-green-500/20 to-emerald-500/20 border border-green-500/30 rounded-xl p-6 hover:border-green-500/50 transition-all duration-300"
                >
                    <div className="relative space-y-2">
                        <h3 className="text-lg font-semibold text-white group-hover:text-green-400 transition-colors">
                            Developer Guide →
                        </h3>
                        <p className="text-slate-400 text-sm">
                            Contribute to the project
                        </p>
                    </div>
                </motion.a>
            </div>

            {/* Platform Support */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.9 }}
                className="bg-gradient-to-r from-slate-800/50 to-slate-800/30 border border-purple-500/20 rounded-xl p-8"
            >
                <h3 className="text-xl font-semibold mb-4 text-center">Cross-Platform Support</h3>
                <div className="flex justify-center gap-8 flex-wrap">
                    <div className="text-center">
                        <div className="w-16 h-16 mx-auto mb-2 bg-slate-700/50 rounded-xl flex items-center justify-center">
                            <span className="text-2xl">🪟</span>
                        </div>
                        <p className="text-sm text-slate-400">Windows</p>
                    </div>
                    <div className="text-center">
                        <div className="w-16 h-16 mx-auto mb-2 bg-slate-700/50 rounded-xl flex items-center justify-center">
                            <span className="text-2xl">🍎</span>
                        </div>
                        <p className="text-sm text-slate-400">macOS</p>
                    </div>
                    <div className="text-center">
                        <div className="w-16 h-16 mx-auto mb-2 bg-slate-700/50 rounded-xl flex items-center justify-center">
                            <span className="text-2xl">🐧</span>
                        </div>
                        <p className="text-sm text-slate-400">Linux</p>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

export default HomeDoc;
