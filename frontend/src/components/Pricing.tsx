import React from "react";
import { Check, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/Button";

export function Pricing() {
    return (
        <section className="py-32 bg-black relative z-20" id="pricing">
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

            <div className="max-w-7xl mx-auto px-6">
                <div className="text-center mb-20">
                    <h2 className="text-4xl md:text-5xl font-heading font-bold mb-6 text-white tracking-tight">
                        Pricing
                    </h2>
                    <p className="text-lg text-neutral-400 max-w-2xl mx-auto">
                        Open source and free forever. No hidden fees.
                    </p>
                </div>

                <div className="flex justify-center">
                    {/* Free Plan */}
                    <div className="w-full max-w-lg rounded-3xl border border-indigo-500/30 bg-neutral-900/50 p-10 flex flex-col relative overflow-hidden shadow-2xl shadow-indigo-500/10 backdrop-blur-sm group hover:border-indigo-500/50 transition-colors duration-300">

                        <div className="absolute top-0 right-0 bg-indigo-600 text-white text-xs font-bold px-4 py-1.5 rounded-bl-xl shadow-lg">
                            OPEN SOURCE
                        </div>

                        <div className="mb-6">
                            <h3 className="text-3xl font-heading font-bold text-white">Community Edition</h3>
                            <p className="text-neutral-400 text-sm mt-2">Everything you need to run local AI</p>
                        </div>

                        <div className="mb-8 flex items-baseline gap-1">
                            <span className="text-6xl font-heading font-bold text-white">$0</span>
                            <span className="text-neutral-500 text-xl font-medium">/forever</span>
                        </div>

                        <ul className="space-y-5 mb-10 flex-1">
                            {[
                                "Full Local LLM Support (Llama 3, Mistral)",
                                "Advanced Code Analysis & Refactoring",
                                "Unlimited Chat History",
                                "Zero Telemetry / 100% Privacy",
                                "Community Plugins & Themes"
                            ].map((feature, i) => (
                                <li key={i} className="flex items-center gap-3 text-neutral-300">
                                    <div className="h-6 w-6 rounded-full bg-indigo-500/20 flex items-center justify-center shrink-0">
                                        <Check className="w-3.5 h-3.5 text-indigo-400" />
                                    </div>
                                    <span className="text-sm font-medium">{feature}</span>
                                </li>
                            ))}
                        </ul>

                        <Button
                            size="lg"
                            className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-lg w-full h-14 rounded-full shadow-lg shadow-indigo-500/25"
                            onClick={() => document.getElementById('download')?.scrollIntoView({ behavior: 'smooth' })}
                        >
                            <div className="flex items-center justify-center gap-2">
                                <span>Download Now</span>
                                <ArrowRight className="w-4 h-4" />
                            </div>
                        </Button>
                    </div>
                </div>
            </div>
        </section>
    );
}
