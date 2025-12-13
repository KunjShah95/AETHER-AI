import { motion } from 'framer-motion';
import { Zap, Cloud, Cpu, Sparkles, MessageSquare, Network } from 'lucide-react';
import CodeBlock from '../CodeBlock';

const ModelsDoc = () => {
    const models = [
        {
            name: 'Gemini 2.0 Flash',
            command: '/switch gemini',
            icon: Sparkles,
            color: 'from-yellow-500 to-orange-500',
            bestFor: 'General tasks, coding, analysis',
            description: "Google's fastest and most capable model with excellent reasoning abilities."
        },
        {
            name: 'Groq Mixtral',
            command: '/switch groq',
            icon: Zap,
            color: 'from-purple-500 to-pink-500',
            bestFor: 'Speed-critical tasks',
            description: 'Ultra-fast inference with incredible performance for real-time applications.'
        },
        {
            name: 'Ollama Local',
            command: '/switch ollama [model]',
            icon: Cpu,
            color: 'from-blue-500 to-cyan-500',
            bestFor: 'Privacy, offline use',
            description: 'Run models locally with complete privacy and no internet dependency.'
        },
        {
            name: 'HuggingFace',
            command: '/switch huggingface',
            icon: Cloud,
            color: 'from-green-500 to-emerald-500',
            bestFor: 'Research, experimentation',
            description: 'Access to thousands of open-source models from the community.'
        },
        {
            name: 'ChatGPT',
            command: '/switch chatgpt',
            icon: MessageSquare,
            color: 'from-teal-500 to-cyan-500',
            bestFor: 'Creative writing, analysis',
            description: "OpenAI's conversational AI with excellent natural language understanding."
        },
        {
            name: 'MCP Protocol',
            command: '/switch mcp',
            icon: Network,
            color: 'from-indigo-500 to-purple-500',
            bestFor: 'Advanced integrations',
            description: 'Model Context Protocol for advanced AI integrations and workflows.'
        },
    ];

    return (
        <div className="space-y-10">
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
            >
                <h1 className="text-4xl lg:text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                    AI Models
                </h1>
                <p className="text-lg text-slate-300">
                    AETHER AI supports a wide range of AI models, allowing you to choose the best tool for the job.
                </p>
            </motion.div>

            {/* Models Grid */}
            <div className="grid md:grid-cols-2 gap-6">
                {models.map((model, index) => {
                    const Icon = model.icon;
                    return (
                        <motion.div
                            key={model.name}
                            initial={{ y: 20, opacity: 0 }}
                            animate={{ y: 0, opacity: 1 }}
                            transition={{ delay: index * 0.1 }}
                            className="group relative overflow-hidden bg-slate-800/30 border border-purple-500/20 rounded-xl p-6 hover:border-purple-500/40 transition-all duration-300"
                        >
                            {/* Background gradient on hover */}
                            <div className={`absolute inset-0 bg-gradient-to-br ${model.color} opacity-0 group-hover:opacity-5 transition-opacity`} />

                            <div className="relative space-y-4">
                                <div className="flex items-start justify-between">
                                    <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${model.color} flex items-center justify-center`}>
                                        <Icon className="w-6 h-6 text-white" />
                                    </div>
                                    <code className="px-3 py-1 bg-slate-900/50 rounded-lg text-xs text-slate-400 font-mono">
                                        {model.command}
                                    </code>
                                </div>

                                <div>
                                    <h3 className="text-xl font-bold text-white mb-1">{model.name}</h3>
                                    <p className="text-sm text-purple-400 font-semibold">{model.bestFor}</p>
                                </div>

                                <p className="text-slate-300 text-sm leading-relaxed">
                                    {model.description}
                                </p>
                            </div>
                        </motion.div>
                    );
                })}
            </div>

            {/* Managing Models */}
            <div className="space-y-6">
                <h2 className="text-2xl font-bold text-white">Managing Models</h2>

                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.6 }}
                    className="space-y-4"
                >
                    <h3 className="text-xl font-semibold text-white">Switching Models</h3>
                    <p className="text-slate-300">
                        You can switch models at any time using the <code className="px-2 py-1 bg-slate-800 rounded text-purple-400">/switch</code> command:
                    </p>

                    <CodeBlock language="bash" code={`/switch gemini
/switch groq
/switch ollama llama3`} />
                </motion.div>

                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.7 }}
                    className="space-y-4"
                >
                    <h3 className="text-xl font-semibold text-white">Checking Status</h3>

                    <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                            <p className="text-sm text-slate-400 mb-2">Check current model:</p>
                            <code className="text-purple-400 font-mono">/current-model</code>
                        </div>
                        <div className="bg-slate-800/30 border border-purple-500/20 rounded-lg p-4">
                            <p className="text-sm text-slate-400 mb-2">List all models:</p>
                            <code className="text-purple-400 font-mono">/models</code>
                        </div>
                    </div>
                </motion.div>
            </div>

            {/* Ollama Integration */}
            <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.8 }}
                className="space-y-6"
            >
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                        <Cpu className="w-6 h-6 text-white" />
                    </div>
                    <h2 className="text-2xl font-bold text-white">Ollama Integration</h2>
                </div>

                <p className="text-slate-300">
                    Ollama allows you to run large language models locally with complete privacy and no API costs.
                </p>

                <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border border-blue-500/30 rounded-xl p-6 space-y-4">
                    <h3 className="font-semibold text-white">Setup Steps:</h3>

                    <ol className="space-y-3">
                        <li className="flex gap-3">
                            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-500 text-white text-sm font-bold shrink-0">
                                1
                            </span>
                            <div>
                                <strong className="text-white">Install Ollama:</strong>
                                <p className="text-slate-300 text-sm mt-1">
                                    Download from <a href="https://ollama.ai" className="text-blue-400 hover:text-blue-300 underline">ollama.ai</a>
                                </p>
                            </div>
                        </li>

                        <li className="flex gap-3">
                            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-500 text-white text-sm font-bold shrink-0">
                                2
                            </span>
                            <div>
                                <strong className="text-white">Pull Models:</strong>
                                <CodeBlock language="bash" code={`ollama pull llama3
ollama pull codellama
ollama pull mistral`} />
                            </div>
                        </li>

                        <li className="flex gap-3">
                            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-500 text-white text-sm font-bold shrink-0">
                                3
                            </span>
                            <div>
                                <strong className="text-white">Use in AETHER:</strong>
                                <div className="mt-2 space-y-2">
                                    <div className="flex items-center gap-2">
                                        <code className="px-2 py-1 bg-slate-800 rounded text-purple-400 text-sm">/ollama-models</code>
                                        <span className="text-slate-400 text-sm">- List available models</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <code className="px-2 py-1 bg-slate-800 rounded text-purple-400 text-sm">/ollama-models [name]</code>
                                        <span className="text-slate-400 text-sm">- View model specs</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <code className="px-2 py-1 bg-slate-800 rounded text-purple-400 text-sm">/switch ollama [name]</code>
                                        <span className="text-slate-400 text-sm">- Switch to local model</span>
                                    </div>
                                </div>
                            </div>
                        </li>
                    </ol>
                </div>
            </motion.div>
        </div>
    );
};

export default ModelsDoc;
