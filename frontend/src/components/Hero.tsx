import React from "react";
import { Button } from "@/components/ui/Button";
import { TerminalWindow } from "@/components/TerminalWindow";
import { ArrowRight, Terminal, Github } from "lucide-react";
import { motion } from "framer-motion";

export function Hero() {
    return (
        <div className="relative min-h-screen w-full flex flex-col items-center justify-center bg-black overflow-hidden pt-20 md:pt-0">
            {/* Subtle Gradient Background */}
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-indigo-900/20 via-black to-black pointer-events-none" />
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150 pointer-events-none mix-blend-overlay"></div>

            <div className="container max-w-7xl mx-auto px-4 md:px-6 relative z-10 flex flex-col md:flex-row items-center gap-12 lg:gap-20">

                {/* Left Content */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, ease: "easeOut" }}
                    className="flex-1 text-center md:text-left space-y-8"
                >
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-indigo-300">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
                        v2.4.0 Release
                    </div>

                    <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-white leading-[1.1]">
                        The AI Assistant <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-blue-400 to-cyan-400">
                            Built for Your Terminal.
                        </span>
                    </h1>

                    <p className="text-lg md:text-xl text-neutral-400 max-w-xl mx-auto md:mx-0 leading-relaxed">
                        Execute complex workflows, audit code security, and switch between LLMs without leaving your CLI. Local-first, secure, and blazing fast.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center gap-4 justify-center md:justify-start">
                        <Button
                            size="lg"
                            variant="gradient"
                            className="w-full sm:w-auto text-base rounded-full h-12 px-8"
                        >
                            <Terminal className="mr-2 h-5 w-5" />
                            Install CLI
                        </Button>
                        <Button
                            size="lg"
                            variant="outline"
                            className="w-full sm:w-auto text-base rounded-full h-12 px-8 border-white/10 bg-white/5 hover:bg-white/10 text-white"
                            onClick={() => window.open('https://github.com/KunjShah95/AETHER-AI.io', '_blank')}
                        >
                            <Github className="mr-2 h-5 w-5" />
                            View Source
                        </Button>
                    </div>

                    <div className="pt-4 flex items-center justify-center md:justify-start gap-6 text-sm text-neutral-500 font-medium">
                        <span className="flex items-center gap-2">
                            <div className="w-1 h-1 rounded-full bg-neutral-500" />
                            Open Source
                        </span>
                        <span className="flex items-center gap-2">
                            <div className="w-1 h-1 rounded-full bg-neutral-500" />
                            Local LLM Support
                        </span>
                        <span className="flex items-center gap-2">
                            <div className="w-1 h-1 rounded-full bg-neutral-500" />
                            No API Key Required
                        </span>
                    </div>
                </motion.div>

                {/* Right Visual */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
                    className="flex-1 w-full max-w-[600px] perspective-[2000px]"
                >
                    <div className="relative transform md:rotate-y-[-5deg] md:rotate-x-[5deg] transition-all duration-500 hover:rotate-0">
                        {/* Glow effect */}
                        <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/5 to-blue-500/5 pointer-events-none blur opacity-20" />
                        <TerminalWindow className="relative bg-black/90 backdrop-blur-xl" />
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
