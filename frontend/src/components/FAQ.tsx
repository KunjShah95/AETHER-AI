import React, { useState } from 'react';
import { Plus, Minus } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const FAQS = [
    {
        question: "Is AetherAI really free?",
        answer: "Yes! The core version of AetherAI is 100% free and open source. You can run it locally with models like Llama 3, Mistral, and Gemma without paying a dime."
    },
    {
        question: "Does my data leave my computer?",
        answer: "No. By default, AetherAI runs entirely on your local machine. Your code, queries, and chat history never leave your device unless you explicitly configure a cloud provider."
    },
    {
        question: "What hardware do I need?",
        answer: "For local inference, we recommend at least 8GB of RAM and a decent CPU (M1/M2/M3 Mac or modern Intel/AMD). For larger models, a dedicated GPU with 6GB+ VRAM is recommended."
    },
    {
        question: "Can I use it with VS Code?",
        answer: "Absolutely. AetherAI integrates directly with your terminal, so it works alongside VS Code, Vim, IntelliJ, or any other editor you prefer."
    },
    {
        question: "How do I install it?",
        answer: "Simply download the installer for your OS from the section above, or run `npm install -g aether-ai` if you prefer the CLI version."
    }
];

export function FAQ() {
    const [openIndex, setOpenIndex] = useState<number | null>(null);

    return (
        <section className="py-24 bg-black relative z-20 border-t border-white/[0.1]">
            <div className="max-w-3xl mx-auto px-6">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-5xl font-bold mb-4 text-white">
                        Frequently Asked Questions
                    </h2>
                    <p className="text-lg text-neutral-400">
                        Everything you need to know about AetherAI.
                    </p>
                </div>

                <div className="space-y-4">
                    {FAQS.map((faq, idx) => (
                        <div key={idx} className="border border-neutral-800 rounded-xl bg-neutral-900/30 overflow-hidden">
                            <button
                                onClick={() => setOpenIndex(openIndex === idx ? null : idx)}
                                className="w-full flex items-center justify-between p-6 text-left hover:bg-neutral-800/50 transition-colors"
                            >
                                <span className="text-lg font-medium text-white">{faq.question}</span>
                                {openIndex === idx ? (
                                    <Minus className="w-5 h-5 text-indigo-400" />
                                ) : (
                                    <Plus className="w-5 h-5 text-neutral-500" />
                                )}
                            </button>
                            <AnimatePresence>
                                {openIndex === idx && (
                                    <motion.div
                                        initial={{ height: 0, opacity: 0 }}
                                        animate={{ height: "auto", opacity: 1 }}
                                        exit={{ height: 0, opacity: 0 }}
                                        transition={{ duration: 0.3 }}
                                    >
                                        <div className="p-6 pt-0 text-neutral-400 leading-relaxed">
                                            {faq.answer}
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
