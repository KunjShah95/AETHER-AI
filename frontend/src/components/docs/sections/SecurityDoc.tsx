import { motion } from 'framer-motion';
import { Shield, Lock, Key, Scan, AlertTriangle, FileKey, RefreshCw } from 'lucide-react';

const SecurityDoc = () => {
    const securityFeatures = [
        {
            title: 'Input Sanitization',
            icon: Scan,
            color: 'from-green-500 to-emerald-500',
            description: 'Blocks dangerous patterns and suspicious Unicode characters to prevent injection attacks'
        },
        {
            title: 'Safe Command Allowlist',
            icon: Shield,
            color: 'from-blue-500 to-cyan-500',
            description: 'The /run command only executes allowlisted commands. No shells, pipes, redirection, or wildcards allowed'
        },
        {
            title: 'API Key Validation',
            icon: Key,
            color: 'from-purple-500 to-pink-500',
            description: 'API keys for major providers are validated for correct format before use'
        },
        {
            title: 'Local Data Storage',
            icon: Lock,
            color: 'from-orange-500 to-red-500',
            description: 'Configuration and user databases stored per-user in ~/.nexus with restricted permissions'
        },
    ];

    const securityCommands = [
        { cmd: '/encrypt [message]', desc: 'Encrypt messages securely', icon: Lock },
        { cmd: '/decrypt [message]', desc: 'Decrypt encrypted messages', icon: FileKey },
        { cmd: '/rotate-key [service] [key]', desc: 'Rotate API keys for services', icon: RefreshCw },
        { cmd: '/biometric-auth [data]', desc: 'Biometric authentication simulation', icon: Scan },
        { cmd: '/secure-password [len]', desc: 'Generate cryptographically secure passwords', icon: Key },
        { cmd: '/security-report', desc: 'View comprehensive security report', icon: Shield },
        { cmd: '/threat-scan [text]', desc: 'Scan text for security threats', icon: AlertTriangle },
    ];

    return (
        <div className="space-y-10">
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
            >
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center">
                        <Shield className="w-8 h-8 text-white" />
                    </div>
                    <div>
                        <h1 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
                            Security
                        </h1>
                        <p className="text-slate-400">Enterprise-grade security built-in</p>
                    </div>
                </div>
                <p className="text-lg text-slate-300">
                    AETHER AI is designed with security as a core priority from the ground up.
                </p>
            </motion.div>

            {/* Security Model */}
            <div className="space-y-4">
                <h2 className="text-2xl font-bold text-white">Security Model</h2>

                <div className="grid md:grid-cols-2 gap-6">
                    {securityFeatures.map((feature, index) => {
                        const Icon = feature.icon;
                        return (
                            <motion.div
                                key={feature.title}
                                initial={{ y: 20, opacity: 0 }}
                                animate={{ y: 0, opacity: 1 }}
                                transition={{ delay: index * 0.1 }}
                                className="group relative overflow-hidden bg-slate-800/30 border border-green-500/20 rounded-xl p-6 hover:border-green-500/40 transition-all duration-300"
                            >
                                {/* Hover effect */}
                                <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-5 transition-opacity`} />

                                <div className="relative space-y-3">
                                    <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center`}>
                                        <Icon className="w-6 h-6 text-white" />
                                    </div>

                                    <h3 className="text-lg font-semibold text-white">{feature.title}</h3>

                                    <p className="text-slate-300 text-sm leading-relaxed">
                                        {feature.description}
                                    </p>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Security Details */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="grid md:grid-cols-2 gap-6"
            >
                <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border border-blue-500/30 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                        <Shield className="w-5 h-5 text-blue-400" />
                        Command Execution Safety
                    </h3>
                    <ul className="space-y-2 text-slate-300 text-sm">
                        <li className="flex items-start gap-2">
                            <span className="text-green-400 shrink-0">✓</span>
                            <span>Allowlist-based command execution</span>
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="text-green-400 shrink-0">✓</span>
                            <span>No shell execution or pipe operations</span>
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="text-green-400 shrink-0">✓</span>
                            <span>File access restricted to current directory</span>
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="text-green-400 shrink-0">✓</span>
                            <span>Automatic argument validation</span>
                        </li>
                    </ul>
                </div>

                <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/30 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                        <Lock className="w-5 h-5 text-purple-400" />
                        Data Protection
                    </h3>
                    <ul className="space-y-2 text-slate-300 text-sm">
                        <li className="flex items-start gap-2">
                            <span className="text-green-400 shrink-0">✓</span>
                            <span>Local-only data storage</span>
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="text-green-400 shrink-0">✓</span>
                            <span>Per-user isolated configuration</span>
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="text-green-400 shrink-0">✓</span>
                            <span>Restricted file permissions</span>
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="text-green-400 shrink-0">✓</span>
                            <span>No telemetry or tracking</span>
                        </li>
                    </ul>
                </div>
            </motion.div>

            {/* Security Commands */}
            <div className="space-y-4">
                <h2 className="text-2xl font-bold text-white">Security Commands</h2>

                <div className="grid gap-3">
                    {securityCommands.map((command, index) => {
                        const Icon = command.icon;
                        return (
                            <motion.div
                                key={command.cmd}
                                initial={{ x: -20, opacity: 0 }}
                                animate={{ x: 0, opacity: 1 }}
                                transition={{ delay: index * 0.05 + 0.5 }}
                                className="group bg-slate-800/30 border border-green-500/20 rounded-lg p-4 hover:border-green-500/40 hover:bg-slate-800/50 transition-all duration-300"
                            >
                                <div className="flex flex-col md:flex-row md:items-center gap-3">
                                    <div className="flex items-center gap-3 md:w-1/3">
                                        <div className="w-8 h-8 rounded-lg bg-green-500/20 flex items-center justify-center shrink-0">
                                            <Icon className="w-4 h-4 text-green-400" />
                                        </div>
                                        <code className="px-3 py-1.5 bg-slate-900/50 border border-green-500/30 rounded-lg text-green-400 font-mono text-xs">
                                            {command.cmd}
                                        </code>
                                    </div>

                                    <p className="text-slate-300 text-sm flex-1">{command.desc}</p>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Best Practices */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.8 }}
                className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-xl p-6"
            >
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-yellow-400" />
                    Security Best Practices
                </h3>
                <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                        <h4 className="font-semibold text-green-400 text-sm">Do:</h4>
                        <ul className="space-y-1 text-slate-300 text-sm">
                            <li className="flex items-start gap-2">
                                <span className="text-green-400 shrink-0">✓</span>
                                <span>Keep your API keys in the .env file</span>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-green-400 shrink-0">✓</span>
                                <span>Regularly rotate your API keys</span>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-green-400 shrink-0">✓</span>
                                <span>Use strong, unique passwords</span>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-green-400 shrink-0">✓</span>
                                <span>Review security reports regularly</span>
                            </li>
                        </ul>
                    </div>
                    <div className="space-y-2">
                        <h4 className="font-semibold text-red-400 text-sm">Don't:</h4>
                        <ul className="space-y-1 text-slate-300 text-sm">
                            <li className="flex items-start gap-2">
                                <span className="text-red-400 shrink-0">✗</span>
                                <span>Share your API keys publicly</span>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-red-400 shrink-0">✗</span>
                                <span>Commit .env files to version control</span>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-red-400 shrink-0">✗</span>
                                <span>Run untrusted commands without review</span>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-red-400 shrink-0">✗</span>
                                <span>Ignore security warnings</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </motion.div>

            {/* Reporting Issues */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.9 }}
                className="bg-gradient-to-br from-yellow-500/10 to-orange-500/10 border border-yellow-500/30 rounded-xl p-6"
            >
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-yellow-400" />
                    Reporting Security Issues
                </h3>
                <p className="text-slate-300 mb-4">
                    If you discover a security vulnerability, please follow responsible disclosure practices:
                </p>
                <ul className="space-y-2 text-slate-300 text-sm">
                    <li className="flex items-start gap-2">
                        <span className="text-yellow-400 shrink-0">1.</span>
                        <span><strong>Do not</strong> disclose the issue publicly until it has been addressed</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <span className="text-yellow-400 shrink-0">2.</span>
                        <span>Refer to <code className="px-2 py-1 bg-slate-800 rounded text-purple-400">SECURITY.md</code> in the repository</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <span className="text-yellow-400 shrink-0">3.</span>
                        <span>Provide detailed information about the vulnerability</span>
                    </li>
                    <li className="flex items-start gap-2">
                        <span className="text-yellow-400 shrink-0">4.</span>
                        <span>Allow reasonable time for the issue to be fixed</span>
                    </li>
                </ul>
            </motion.div>
        </div>
    );
};

export default SecurityDoc;
