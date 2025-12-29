import React from "react";
import {
    Brain,
    Terminal,
    CheckCircle2,
    Code2,
    Palette,
    Shield,
    Workflow,
    Cpu
} from "lucide-react";

const features = [
    {
        title: "Model Agnostic",
        description: "Switch seamlessly between Gemini, Groq, DeepSeek, and local Ollama models with a single command.",
        icon: Brain,
        color: "text-indigo-400"
    },
    {
        title: "Native TUI",
        description: "A rich Terminal User Interface with syntax highlighting, mouse support, and vim-keybindings.",
        icon: Terminal,
        color: "text-cyan-400"
    },
    {
        title: "Code Analysis",
        description: "Instant security audits and performance profiling for 15+ languages directly in your workflow.",
        icon: Code2,
        color: "text-blue-400"
    },
    {
        title: "Local First",
        description: "Your code never leaves your machine. Full privacy by default with optional cloud processing.",
        icon: Shield,
        color: "text-emerald-400"
    },
    {
        title: "Workflow Automation",
        description: "Chain agents together to perform complex tasks: search -> analyze -> refactor -> commit.",
        icon: Workflow,
        color: "text-amber-400"
    },
    {
        title: "Custom Theming",
        description: "Match your assistant to your system. JSON-based themes with 6 premium presets included.",
        icon: Palette,
        color: "text-teal-400"
    }
];

export function Features() {
    return (
        <section id="features" className="py-24 md:py-32 bg-black relative overflow-hidden">
            {/* Background Elements */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
            <div className="absolute left-0 right-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

            <div className="container max-w-7xl mx-auto px-6 relative z-10">
                <div className="text-center max-w-3xl mx-auto mb-20 space-y-4">
                    <h2 className="text-3xl md:text-5xl font-bold tracking-tight text-white">
                        Built for the
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400 ml-2">
                            Modern CLI
                        </span>
                    </h2>
                    <p className="text-lg text-neutral-400 leading-relaxed">
                        AETHER AI isn't just a wrapper. It's a comprehensive operating system for your terminal,
                        designed to make you 10x more productive while keeping your hands on the keyboard.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {features.map((feature, idx) => (
                        <div
                            key={idx}
                            className="group p-8 rounded-2xl bg-white/[0.03] border border-white/10 hover:bg-white/[0.05] hover:border-white/20 transition-all duration-300"
                        >
                            <div className="mb-6 inline-flex p-3 rounded-lg bg-white/5 ring-1 ring-white/10 group-hover:scale-110 transition-transform duration-300">
                                <feature.icon className={`w-6 h-6 ${feature.color}`} />
                            </div>

                            <h3 className="text-xl font-semibold text-white mb-3 flex items-center gap-2">
                                {feature.title}
                            </h3>

                            <p className="text-neutral-400 leading-relaxed text-sm">
                                {feature.description}
                            </p>
                        </div>
                    ))}
                </div>

                <div className="mt-20 p-8 rounded-2xl bg-gradient-to-r from-indigo-900/20 to-blue-900/20 border border-indigo-500/20 text-center">
                    <div className="flex flex-col md:flex-row items-center justify-center gap-8">
                        <div className="text-left space-y-1">
                            <div className="text-2xl font-bold text-white">100% Open Source</div>
                            <div className="text-sm text-neutral-400">Join the community on GitHub</div>
                        </div>
                        <div className="h-10 w-px bg-white/10 hidden md:block" />
                        <div className="text-left space-y-1">
                            <div className="text-2xl font-bold text-white">15+ Languages</div>
                            <div className="text-sm text-neutral-400">Supported out of the box</div>
                        </div>
                        <div className="h-10 w-px bg-white/10 hidden md:block" />
                        <div className="text-left space-y-1">
                            <div className="text-2xl font-bold text-white">Local LLCs</div>
                            <div className="text-sm text-neutral-400">Run generic models offline</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}
