import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Terminal, Maximize2, Minus, X } from 'lucide-react';

export const TerminalWindow = ({ className }: { className?: string }) => {
    const [lines, setLines] = useState([
        { type: 'command', content: 'nexus init --mode=production' },
        { type: 'output', content: 'Initializing NEXUS engine v2.4.0...' },
        { type: 'output', content: '✓ Neural Core loaded (20ms)' },
        { type: 'output', content: '✓ Security Protocols active' },
        { type: 'output', content: '✓ Connected to local LLM cluster' },
        { type: 'command', content: 'nexus analyze --target=./src' },
        { type: 'output', content: 'Scanning codebase for vulnerabilities...' },
    ]);

    return (
        <div className={cn("rounded-lg overflow-hidden border border-white/10 bg-[#0c0c0c] shadow-2xl font-mono text-sm", className)}>
            {/* Window Controls */}
            <div className="bg-[#1a1a1a] px-4 py-2 flex items-center justify-between border-b border-white/5">
                <div className="flex gap-2">
                    <div className="w-3 h-3 rounded-full bg-[#FF5F56] hover:bg-[#FF5F56]/80 transition-colors" />
                    <div className="w-3 h-3 rounded-full bg-[#FFBD2E] hover:bg-[#FFBD2E]/80 transition-colors" />
                    <div className="w-3 h-3 rounded-full bg-[#27C93F] hover:bg-[#27C93F]/80 transition-colors" />
                </div>
                <div className="text-neutral-500 text-xs flex items-center gap-1.5 opacity-50">
                    <Terminal size={10} />
                    <span>user@nexus-ai:~</span>
                </div>
                <div className="w-10" /> {/* Spacer for centering */}
            </div>

            {/* Terminal Content */}
            <div className="p-4 space-y-2 min-h-[300px] text-neutral-300">
                {lines.map((line, idx) => (
                    <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.15 }}
                        className="flex gap-2"
                    >
                        {line.type === 'command' ? (
                            <>
                                <span className="text-emerald-400 font-bold">➜</span>
                                <span className="text-cyan-400 font-bold">~</span>
                                <span className="text-white">{line.content}</span>
                            </>
                        ) : (
                            <span className={cn(
                                "pl-6",
                                line.content.includes('✓') ? "text-emerald-400" : "text-neutral-400"
                            )}>
                                {line.content}
                            </span>
                        )}
                    </motion.div>
                ))}

                {/* Active Prompt */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: lines.length * 0.15 + 0.5 }}
                    className="flex gap-2 items-center"
                >
                    <span className="text-emerald-400 font-bold">➜</span>
                    <span className="text-cyan-400 font-bold">~</span>
                    <span className="w-2 h-4 bg-white animate-pulse" />
                </motion.div>
            </div>
        </div>
    );
};
