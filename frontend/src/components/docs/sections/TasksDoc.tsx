import { motion } from 'framer-motion';
import { CheckSquare, Plus, List, Eye, Edit, Trash2, Star, Calendar, Tag, Filter } from 'lucide-react';
import CodeBlock from '../CodeBlock';

const TasksDoc = () => {
    const basicOps = [
        { cmd: '/task add [title] [description]', desc: 'Add a new task', example: '/task add "Fix bug" "Login issue"', icon: Plus },
        { cmd: '/task list', desc: 'List all tasks', icon: List },
        { cmd: '/task view [id]', desc: 'View task details', example: '/task view 1', icon: Eye },
        { cmd: '/task update [id] [field] [value]', desc: 'Update task field', example: '/task update 1 status completed', icon: Edit },
        { cmd: '/task delete [id]', desc: 'Delete a task', example: '/task delete 1', icon: Trash2 },
        { cmd: '/task complete [id]', desc: 'Mark task as completed', example: '/task complete 1', icon: CheckSquare },
    ];

    const advancedFeatures = [
        {
            title: 'Priority Levels',
            icon: Star,
            color: 'from-yellow-500 to-orange-500',
            description: 'Set task priority to organize your workload',
            command: '/task priority [id] [level]',
            example: '/task priority 1 high'
        },
        {
            title: 'Categories',
            icon: Tag,
            color: 'from-purple-500 to-pink-500',
            description: 'Organize tasks by category for better management',
            command: '/task category [id] [category]',
            example: '/task category 1 development'
        },
        {
            title: 'Due Dates',
            icon: Calendar,
            color: 'from-blue-500 to-cyan-500',
            description: 'Set deadlines to stay on track',
            command: '/task due [id] [YYYY-MM-DD]',
            example: '/task due 1 2024-12-31'
        },
        {
            title: 'Subtasks',
            icon: List,
            color: 'from-green-500 to-emerald-500',
            description: 'Break down complex tasks into manageable pieces',
            command: '/task subtasks [id] [subtask]',
            example: '/task subtasks 1 "Research solution"'
        },
        {
            title: 'Search & Filter',
            icon: Filter,
            color: 'from-indigo-500 to-purple-500',
            description: 'Quickly find what you need',
            command: '/task search [keyword]',
            example: '/task search bug'
        },
    ];

    return (
        <div className="space-y-10">
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
            >
                <h1 className="text-4xl lg:text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                    Task Management System
                </h1>
                <p className="text-lg text-slate-300">
                    Stay organized and productive with AETHER AI's built-in task management system, accessible directly from your terminal.
                </p>
            </motion.div>

            {/* Feature Highlights */}
            <div className="grid md:grid-cols-3 gap-4">
                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.1 }}
                    className="bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-500/30 rounded-xl p-6"
                >
                    <div className="text-3xl mb-2">✅</div>
                    <h3 className="font-semibold text-white mb-1">Simple & Fast</h3>
                    <p className="text-sm text-slate-300">Add and manage tasks without leaving your terminal</p>
                </motion.div>

                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border border-blue-500/30 rounded-xl p-6"
                >
                    <div className="text-3xl mb-2">🎯</div>
                    <h3 className="font-semibold text-white mb-1">Powerful Features</h3>
                    <p className="text-sm text-slate-300">Priorities, categories, due dates, and more</p>
                </motion.div>

                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 border border-green-500/30 rounded-xl p-6"
                >
                    <div className="text-3xl mb-2">📊</div>
                    <h3 className="font-semibold text-white mb-1">Track Progress</h3>
                    <p className="text-sm text-slate-300">View statistics and export data anytime</p>
                </motion.div>
            </div>

            {/* Basic Operations */}
            <div className="space-y-4">
                <h2 className="text-2xl font-bold text-white">Basic Operations</h2>

                <div className="grid gap-3">
                    {basicOps.map((op, index) => {
                        const Icon = op.icon;
                        return (
                            <motion.div
                                key={op.cmd}
                                initial={{ x: -20, opacity: 0 }}
                                animate={{ x: 0, opacity: 1 }}
                                transition={{ delay: index * 0.05 + 0.4 }}
                                className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4 hover:border-purple-500/40 hover:bg-slate-800/50 transition-all duration-300"
                            >
                                <div className="flex flex-col md:flex-row md:items-center gap-3">
                                    <div className="flex items-center gap-3 md:w-1/3">
                                        <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center shrink-0">
                                            <Icon className="w-4 h-4 text-purple-400" />
                                        </div>
                                        <code className="px-3 py-1.5 bg-slate-900/50 border border-purple-500/30 rounded-lg text-purple-400 font-mono text-xs">
                                            {op.cmd}
                                        </code>
                                    </div>

                                    <div className="flex-1">
                                        <p className="text-slate-300 text-sm">{op.desc}</p>
                                        {op.example && (
                                            <code className="block mt-1 text-xs text-green-400">
                                                {op.example}
                                            </code>
                                        )}
                                    </div>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Advanced Features */}
            <div className="space-y-4">
                <h2 className="text-2xl font-bold text-white">Advanced Features</h2>

                <div className="grid md:grid-cols-2 gap-6">
                    {advancedFeatures.map((feature, index) => {
                        const Icon = feature.icon;
                        return (
                            <motion.div
                                key={feature.title}
                                initial={{ y: 20, opacity: 0 }}
                                animate={{ y: 0, opacity: 1 }}
                                transition={{ delay: index * 0.1 + 0.7 }}
                                className="bg-slate-800/30 border border-purple-500/20 rounded-xl p-6 hover:border-purple-500/40 transition-all duration-300"
                            >
                                <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4`}>
                                    <Icon className="w-5 h-5 text-white" />
                                </div>

                                <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                                <p className="text-slate-300 text-sm mb-3">{feature.description}</p>

                                <div className="space-y-2">
                                    <code className="block px-3 py-1.5 bg-slate-900/50 border border-purple-500/30 rounded-lg text-purple-400 font-mono text-xs">
                                        {feature.command}
                                    </code>
                                    <code className="block text-xs text-green-400">
                                        Example: {feature.example}
                                    </code>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Tools */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 1.2 }}
                className="space-y-4"
            >
                <h2 className="text-2xl font-bold text-white">Additional Tools</h2>

                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                        <h4 className="font-semibold text-white mb-2 text-sm">Export/Import</h4>
                        <div className="space-y-1 text-xs">
                            <code className="block text-purple-400">/task export [json/csv]</code>
                            <code className="block text-purple-400">/task import [file]</code>
                        </div>
                    </div>

                    <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                        <h4 className="font-semibold text-white mb-2 text-sm">Statistics</h4>
                        <div className="space-y-1 text-xs">
                            <code className="block text-purple-400">/task stats</code>
                            <code className="block text-purple-400">/task categories</code>
                        </div>
                    </div>

                    <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                        <h4 className="font-semibold text-white mb-2 text-sm">Quick Views</h4>
                        <div className="space-y-1 text-xs">
                            <code className="block text-purple-400">/task overdue</code>
                            <code className="block text-purple-400">/task today</code>
                        </div>
                    </div>

                    <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                        <h4 className="font-semibold text-white mb-2 text-sm">Filtering</h4>
                        <div className="space-y-1 text-xs">
                            <code className="block text-purple-400">/task search [keyword]</code>
                            <code className="block text-purple-400">/task filter [criteria]</code>
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* Example Workflow */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 1.3 }}
                className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-xl p-6"
            >
                <h3 className="text-lg font-semibold text-white mb-3">💡 Example Workflow</h3>
                <CodeBlock language="bash" code={`# Add a new task
/task add "Implement login feature" "Add OAuth2 authentication"

# Set priority and category
/task priority 1 high
/task category 1 development

# Set due date
/task due 1 2024-12-31

# Add subtasks
/task subtasks 1 "Setup OAuth provider"
/task subtasks 1 "Create login UI"
/task subtasks 1 "Add tests"

# View all tasks
/task list

# Mark as complete
/task complete 1`} />
            </motion.div>
        </div>
    );
};

export default TasksDoc;
