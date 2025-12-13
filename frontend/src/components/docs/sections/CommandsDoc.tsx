import { motion } from 'framer-motion';
import { Terminal, Wrench, Code, GitBranch, Search } from 'lucide-react';

const CommandsDoc = () => {
    const commandCategories = [
        {
            title: 'System Commands',
            icon: Terminal,
            color: 'from-purple-500 to-blue-500',
            commands: [
                { cmd: '/help', desc: 'Show comprehensive help menu' },
                { cmd: '/status', desc: 'Display current model and system status' },
                { cmd: '/clear', desc: 'Clear the terminal screen' },
                { cmd: '/exit', desc: 'Exit the AETHER AI terminal' },
                { cmd: '/sysinfo', desc: 'Show detailed system information' },
                { cmd: '/config', desc: 'Display current configuration' },
            ]
        },
        {
            title: 'Utility Commands',
            icon: Wrench,
            color: 'from-blue-500 to-cyan-500',
            commands: [
                { cmd: '/run [command]', desc: 'Execute safe system commands', example: '/run ls -la' },
                { cmd: '/calc [expression]', desc: 'Calculate mathematical expressions', example: '/calc 2 + 2 * 3' },
                { cmd: '/websearch [query]', desc: 'Search the web using DuckDuckGo', example: '/websearch python tutorials' },
                { cmd: '/weather [city]', desc: 'Get current weather information', example: '/weather New York' },
                { cmd: '/note [text]', desc: 'Save a quick note', example: '/note Buy milk' },
                { cmd: '/notes', desc: 'View all saved notes' },
                { cmd: '/timer [seconds]', desc: 'Start a countdown timer', example: '/timer 300' },
                { cmd: '/convert [val] [from] [to]', desc: 'Unit converter', example: '/convert 100 celsius fahrenheit' },
                { cmd: '/password [length]', desc: 'Generate secure password', example: '/password 16' },
            ]
        },
        {
            title: 'Developer Commands',
            icon: Code,
            color: 'from-green-500 to-emerald-500',
            commands: [
                { cmd: '/codereview [filename]', desc: 'AI code review for bugs and improvements' },
                { cmd: '/summarizefile [filename]', desc: 'AI file summarization' },
                { cmd: '/findbugs [filename]', desc: 'Find bugs in code using AI' },
                { cmd: '/refactor [filename] [instruction]', desc: 'AI code refactoring' },
                { cmd: '/gendoc [filename]', desc: 'Generate documentation for code' },
                { cmd: '/gentest [filename]', desc: 'Generate unit tests for code' },
                { cmd: '/git commitmsg [diff]', desc: 'Generate git commit messages' },
                { cmd: '/todos', desc: 'Extract TODOs and FIXMEs from codebase' },
            ]
        },
        {
            title: 'Git Advanced Commands',
            icon: GitBranch,
            color: 'from-orange-500 to-red-500',
            commands: [
                { cmd: '/git create-branch [name]', desc: 'Create a new branch' },
                { cmd: '/git delete-branch [name]', desc: 'Delete a branch' },
                { cmd: '/aifind [keyword]', desc: 'AI-powered file search' },
                { cmd: '/explore', desc: 'Explore codebase' },
            ]
        },
        {
            title: 'Advanced AI Features',
            icon: Code,
            color: 'from-pink-500 to-rose-500',
            commands: [
                { cmd: '/persona [name]', desc: 'Switch AI persona' },
                { cmd: '/web [url]', desc: 'Web agent browse' },
                { cmd: '/docker init', desc: 'Dockerize project' },
                { cmd: '/dashboard', desc: 'Open Mission Control' },
                { cmd: '/voice', desc: 'Toggle Voice Mode' },
            ]
        }
    ];

    return (
        <div className="space-y-10">
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
            >
                <h1 className="text-4xl lg:text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                    Commands Reference
                </h1>
                <p className="text-lg text-slate-300">
                    Complete guide to all available commands in AETHER AI Terminal Assistant.
                </p>
            </motion.div>

            {/* Search Box */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.1 }}
                className="relative"
            >
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
                <input
                    type="text"
                    placeholder="Search commands..."
                    className="w-full pl-12 pr-4 py-3 bg-slate-800/50 border border-purple-500/20 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                />
            </motion.div>

            {/* Command Categories */}
            {commandCategories.map((category, categoryIndex) => {
                const Icon = category.icon;

                return (
                    <motion.div
                        key={category.title}
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ delay: categoryIndex * 0.1 + 0.2 }}
                        className="space-y-4"
                    >
                        <div className="flex items-center gap-3">
                            <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${category.color} flex items-center justify-center`}>
                                <Icon className="w-6 h-6 text-white" />
                            </div>
                            <h2 className="text-2xl font-bold text-white">{category.title}</h2>
                        </div>

                        <div className="space-y-3">
                            {category.commands.map((command, cmdIndex) => (
                                <motion.div
                                    key={command.cmd}
                                    initial={{ x: -20, opacity: 0 }}
                                    animate={{ x: 0, opacity: 1 }}
                                    transition={{ delay: categoryIndex * 0.1 + cmdIndex * 0.05 + 0.3 }}
                                    className="group bg-slate-800/30 border border-purple-500/20 rounded-lg p-4 hover:border-purple-500/40 hover:bg-slate-800/50 transition-all duration-300"
                                >
                                    <div className="flex flex-col md:flex-row md:items-start gap-3">
                                        <code className="px-3 py-1.5 bg-slate-900/50 border border-purple-500/30 rounded-lg text-purple-400 font-mono text-sm shrink-0">
                                            {command.cmd}
                                        </code>

                                        <div className="flex-1 min-w-0">
                                            <p className="text-slate-300">{command.desc}</p>
                                            {command.example && (
                                                <div className="mt-2">
                                                    <span className="text-xs text-slate-500 uppercase tracking-wide">Example:</span>
                                                    <code className="block mt-1 text-xs text-green-400 font-mono">
                                                        {command.example}
                                                    </code>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>
                );
            })}

            {/* Quick Tip */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.8 }}
                className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-xl p-6"
            >
                <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                    💡 Pro Tip
                </h3>
                <p className="text-slate-300">
                    Type <code className="px-2 py-1 bg-slate-800 rounded text-purple-400">/help</code> in the terminal
                    to see a comprehensive, interactive help menu with all available commands and their descriptions.
                </p>
            </motion.div>
        </div>
    );
};

export default CommandsDoc;
