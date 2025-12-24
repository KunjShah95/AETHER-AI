import React from "react";
import { Laptop, Server, ShieldCheck, ArrowRight } from "lucide-react";

export function HowItWorks() {
    return (
        <section className="py-32 bg-black relative z-20">
            <div className="absolute inset-0 bg-grid-white/[0.02] pointer-events-none" />

            <div className="max-w-7xl mx-auto px-6 relative z-10">
                <div className="text-center mb-24">
                    <h2 className="text-4xl md:text-5xl font-heading font-bold mb-6 text-white tracking-tight">
                        Architecture of <span className="text-indigo-500">Privacy</span>
                    </h2>
                    <p className="text-lg text-neutral-400 max-w-2xl mx-auto leading-relaxed">
                        A local-first pipeline designed to keep your intellectual property safe while delivering cloud-grade intelligence.
                    </p>
                </div>

                <div className="flex flex-col md:flex-row items-start justify-center gap-8 relative">

                    {/* Connecting Line (Pro) */}
                    <div className="absolute top-12 left-[15%] right-[15%] h-[2px] bg-gradient-to-r from-indigo-500/0 via-indigo-500/50 to-indigo-500/0 hidden md:block -z-10" />

                    {/* Step 1 */}
                    <div className="flex flex-col items-center text-center max-w-xs relative group mx-auto md:mx-0">
                        <div className="w-24 h-24 rounded-2xl bg-neutral-900 border border-white/10 flex items-center justify-center mb-8 shadow-2xl shadow-indigo-500/10 group-hover:shadow-indigo-500/30 group-hover:scale-105 transition-all duration-300 relative overflow-hidden">
                            <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                            <Laptop className="w-10 h-10 text-indigo-400 relative z-10" />
                        </div>
                        <h3 className="text-xl font-heading font-bold text-white mb-3">1. Install Core</h3>
                        <p className="text-neutral-400 text-sm leading-relaxed px-4">
                            One-line install via pip. The lightweight daemon runs in the background, consuming minimal resources.
                        </p>
                    </div>

                    {/* Step 2 */}
                    <div className="flex flex-col items-center text-center max-w-xs relative group mx-auto md:mx-0 mt-8 md:mt-0">
                        <div className="w-24 h-24 rounded-2xl bg-neutral-900 border border-white/10 flex items-center justify-center mb-8 shadow-2xl shadow-blue-500/10 group-hover:shadow-blue-500/30 group-hover:scale-105 transition-all duration-300 relative overflow-hidden">
                            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                            <Server className="w-10 h-10 text-blue-400 relative z-10" />
                        </div>
                        <h3 className="text-xl font-heading font-bold text-white mb-3">2. Choose Engine</h3>
                        <p className="text-neutral-400 text-sm leading-relaxed px-4">
                            Select `ollama` for 100% offline privacy, or connect to `groq`/`gemini` for massive context windows.
                        </p>
                    </div>

                    {/* Step 3 */}
                    <div className="flex flex-col items-center text-center max-w-xs relative group mx-auto md:mx-0 mt-8 md:mt-0">
                        <div className="w-24 h-24 rounded-2xl bg-neutral-900 border border-white/10 flex items-center justify-center mb-8 shadow-2xl shadow-emerald-500/10 group-hover:shadow-emerald-500/30 group-hover:scale-105 transition-all duration-300 relative overflow-hidden">
                            <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                            <ShieldCheck className="w-10 h-10 text-emerald-400 relative z-10" />
                        </div>
                        <h3 className="text-xl font-heading font-bold text-white mb-3">3. Secure Workflow</h3>
                        <p className="text-neutral-400 text-sm leading-relaxed px-4">
                            All context retrieval (RAG) happens locally. We sanitize inputs before they ever touch a model.
                        </p>
                    </div>
                </div>
            </div>
        </section>
    );
}
