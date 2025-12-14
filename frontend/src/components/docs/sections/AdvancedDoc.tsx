import { motion } from 'framer-motion';
import { Brain, BarChart3, Gamepad2, Palette, Users } from 'lucide-react';

const AdvancedDoc = () => {
    const features = [
        {
            title: 'Context-Aware AI',
            icon: Brain,
            color: 'from-purple-500 to-blue-500',
            commands: [
                { cmd: '/learn [topic]', desc: 'Teach AI about specific technologies or topics' },
                { cmd: '/remind [task]', desc: 'Set a new reminder' },
                { cmd: '/reminders', desc: 'View all reminders' },
                { cmd: '/complete-reminder [n]', desc: 'Mark reminder as complete' },
            ]
        },
        {
            title: 'Autonomous Agents',
            icon: Brain,
            color: 'from-pink-500 to-rose-500',
            commands: [
                { cmd: '/persona [name]', desc: 'Switch expert persona (security, qa)' },
                { cmd: '/web [url]', desc: 'Autonomous web agent research' },
                { cmd: '/docker init', desc: 'Auto-generate Docker environment' },
                { cmd: '/autopilot [goal]', desc: 'Autonomous workflow execution' },
            ]
        },
        {
            title: 'Analytics & Monitoring',
            icon: BarChart3,
            color: 'from-blue-500 to-cyan-500',
            commands: [
                { cmd: '/analytics', desc: 'View usage statistics' },
                { cmd: '/error-analytics', desc: 'View error analytics' },
                { cmd: '/start-monitoring', desc: 'Start system monitoring' },
                { cmd: '/stop-monitoring', desc: 'Stop system monitoring' },
                { cmd: '/net-diag', desc: 'Network diagnostics' },
                { cmd: '/analyze-logs', desc: 'Analyze log files' },
                { cmd: '/health', desc: 'System health check' },
            ]
        },
        {
            title: 'Games & Learning',
            icon: Gamepad2,
            color: 'from-green-500 to-emerald-500',
            commands: [
                { cmd: '/challenge [difficulty]', desc: 'Start a coding challenge' },
                { cmd: '/tutorial [topic]', desc: 'Interactive tutorial' },
                { cmd: '/quiz [topic]', desc: 'Take a knowledge quiz' },
                { cmd: '/user-stats', desc: 'View your learning statistics' },
            ]
        },
        {
            title: 'Creative Tools',
            icon: Palette,
            color: 'from-pink-500 to-rose-500',
            commands: [
                { cmd: '/ascii [text]', desc: 'Generate ASCII art' },
                { cmd: '/colors [type] [base]', desc: 'Generate color schemes' },
                { cmd: '/music [mood] [length]', desc: 'Generate music patterns' },
                { cmd: '/story [genre] [length]', desc: 'Generate creative stories' },
            ]
        },
        {
            title: 'User Management',
            icon: Users,
            color: 'from-orange-500 to-red-500',
            commands: [
                { cmd: '/setkey [provider] [key]', desc: 'Set API keys' },
                { cmd: '/history', desc: 'View chat history' },
                { cmd: '/clearhistory', desc: 'Clear chat history' },
                { cmd: '/myactivity', desc: 'View activity log' },
                { cmd: '/listusers', desc: 'List all users (admin)' },
                { cmd: '/resetpw [user] [newpass]', desc: 'Reset password (admin)' },
            ]
        },
    ];

    return (
        <div className="space-y-10">
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
            >
                <h1 className="text-4xl lg:text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                    Advanced Features
                </h1>
                <p className="text-lg text-slate-300">
                    Unlock the full power of AETHER AI with advanced features for productivity, creativity, and development.
                </p>
            </motion.div>

            {/* Features */}
            {features.map((feature, index) => {
                const Icon = feature.icon;

                return (
                    <motion.div
                        key={feature.title}
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ delay: index * 0.1 }}
                        className="space-y-4"
                    >
                        <div className="flex items-center gap-3">
                            <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center`}>
                                <Icon className="w-6 h-6 text-white" />
                            </div>
                            <h2 className="text-2xl font-bold text-white">{feature.title}</h2>
                        </div>

                        <div className="grid md:grid-cols-2 gap-4">
                            {feature.commands.map((command, cmdIndex) => (
                                <motion.div
                                    key={command.cmd}
                                    initial={{ x: -20, opacity: 0 }}
                                    animate={{ x: 0, opacity: 1 }}
                                    transition={{ delay: index * 0.1 + cmdIndex * 0.05 }}
                                    className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4 hover:border-purple-500/40 hover:bg-slate-800/50 transition-all duration-300"
                                >
                                    <code className="block mb-2 px-3 py-1.5 bg-slate-900/50 border border-purple-500/30 rounded-lg text-purple-400 font-mono text-sm w-fit">
                                        {command.cmd}
                                    </code>
                                    <p className="text-slate-300 text-sm">{command.desc}</p>
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>
                );
            })}

            {/* Highlight Box */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.8 }}
                className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-xl p-6"
            >
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                    🚀 Pro Features
                </h3>
                <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                        <h4 className="font-semibold text-purple-400">Context Learning</h4>
                        <p className="text-sm text-slate-300">
                            Teach the AI about your specific tech stack, projects, or domain knowledge for more relevant responses.
                        </p>
                    </div>
                    <div className="space-y-2">
                        <h4 className="font-semibold text-blue-400">Performance Monitoring</h4>
                        <p className="text-sm text-slate-300">
                            Track system performance, analyze logs, and get real-time diagnostics for optimal operation.
                        </p>
                    </div>
                    <div className="space-y-2">
                        <h4 className="font-semibold text-green-400">Interactive Learning</h4>
                        <p className="text-sm text-slate-300">
                            Improve your coding skills with challenges, tutorials, and quizzes tailored to your level.
                        </p>
                    </div>
                    <div className="space-y-2">
                        <h4 className="font-semibold text-pink-400">Creative Tools</h4>
                        <p className="text-sm text-slate-300">
                            Generate ASCII art, color schemes, music patterns, and creative content directly from your terminal.
                        </p>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

export default AdvancedDoc;
