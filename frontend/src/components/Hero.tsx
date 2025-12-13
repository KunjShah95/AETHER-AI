import React from "react";
import { Spotlight } from "@/components/ui/Spotlight";
import { Button } from "@/components/ui/MovingBorder";
import { Play, ArrowRight } from "lucide-react";

export function Hero() {
    return (
        <div className="min-h-[90vh] w-full rounded-md flex md:items-center md:justify-center bg-black/[0.96] antialiased bg-grid-white/[0.02] relative overflow-hidden">
            <Spotlight
                className="-top-40 left-0 md:left-60 md:-top-20"
                fill="white"
            />

            <div className="p-4 max-w-7xl mx-auto relative z-10 w-full pt-20 md:pt-0 flex flex-col md:flex-row items-center justify-between gap-12">

                {/* Left Side: Text Content */}
                <div className="flex-1 text-center md:text-left z-20">
                    <h1 className="text-4xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400 bg-opacity-50 leading-tight pb-4">
                        Master Your Workflow <br />
                        <span className="text-terminal-cyan">With Local AI.</span>
                    </h1>

                    <p className="mt-4 font-normal text-base text-neutral-300 max-w-lg mx-auto md:mx-0 leading-relaxed">
                        Experience the power of a fully offline, privacy-focused AI assistant.
                        No latency, no data leaks, just pure productivity in your terminal.
                    </p>

                    <div className="mt-10 flex flex-col md:flex-row gap-4 justify-center md:justify-start items-center">
                        <Button
                            borderRadius="1.75rem"
                            className="bg-slate-900 text-white border-slate-800 font-bold"
                            onClick={() => document.getElementById('download')?.scrollIntoView({ behavior: 'smooth' })}
                        >
                            <div className="flex items-center gap-2">
                                <span>Get Started Free</span>
                                <ArrowRight className="w-4 h-4" />
                            </div>
                        </Button>

                        <button
                            onClick={() => window.open('https://www.youtube.com/watch?v=DLyrTzcYgbI', '_blank')}
                            className="px-8 py-3 rounded-full bg-transparent border border-neutral-700 text-neutral-300 hover:bg-neutral-800 transition-colors flex items-center gap-2 font-medium"
                        >
                            <Play className="w-4 h-4" />
                            <span>Watch Demo</span>
                        </button>
                    </div>

                    <div className="mt-8 flex items-center justify-center md:justify-start gap-4 text-sm text-neutral-500">
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-green-500"></div>
                            <span>v4.0.0 Stable</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                            <span>macOS, Windows, Linux</span>
                        </div>
                    </div>
                </div>

                {/* Right Side: Dashboard Image */}
                <div className="flex-1 w-full max-w-2xl relative z-10">
                    <div className="relative rounded-xl overflow-hidden border border-white/10 shadow-2xl shadow-purple-500/20 group">
                        <div className="absolute inset-0 bg-gradient-to-tr from-purple-500/10 to-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                        <img
                            src="/nexus_ai_dashboard.svg"
                            alt="AETHER-AI Dashboard - Interactive Terminal with Neural Network Visualization"
                            className="w-full h-auto object-cover transform transition-transform duration-700 hover:scale-105"
                        />
                        {/* Overlay for better text contrast if needed, or just aesthetic sheen */}
                        <div className="absolute inset-0 bg-black/20 group-hover:bg-black/0 transition-colors duration-500"></div>
                    </div>

                    {/* Decorative Elements behind the image */}
                    <div className="absolute -bottom-10 -right-10 w-72 h-72 bg-purple-500/30 rounded-full blur-3xl -z-10"></div>
                    <div className="absolute -top-10 -left-10 w-72 h-72 bg-cyan-500/30 rounded-full blur-3xl -z-10"></div>
                </div>

            </div>
        </div>
    );
}
