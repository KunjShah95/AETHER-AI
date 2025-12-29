import React, { useState, useEffect, useRef } from 'react';
import { Terminal as TerminalIcon, Maximize2, Minus, X } from 'lucide-react';

const COMMANDS = [
    { cmd: 'aether init', output: 'Initializing Aether AI environment...' },
    { cmd: 'aether connect --model=llama-3', output: 'Connected to local Llama 3 instance. Ready.' },
    { cmd: 'aether analyze src/App.tsx', output: 'Analyzing App.tsx...\nFound 2 potential optimizations.\nSecurity check passed.' },
    { cmd: 'aether chat "How do I optimize this?"', output: 'To optimize this component, consider memoizing the callback functions...' },
];

export function Terminal() {
    const [lines, setLines] = useState<Array<{ type: 'cmd' | 'output', text: string }>>([]);
    const [currentCommandIndex, setCurrentCommandIndex] = useState(0);
    const [currentText, setCurrentText] = useState('');
    const [isTyping, setIsTyping] = useState(true);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (currentCommandIndex >= COMMANDS.length) {
            // Reset after a delay
            const timeout = setTimeout(() => {
                setLines([]);
                setCurrentCommandIndex(0);
                setCurrentText('');
                setIsTyping(true);
            }, 5000);
            return () => clearTimeout(timeout);
        }

        const command = COMMANDS[currentCommandIndex];

        if (isTyping) {
            if (currentText.length < command.cmd.length) {
                const timeout = setTimeout(() => {
                    setCurrentText(command.cmd.slice(0, currentText.length + 1));
                }, 50 + Math.random() * 50);
                return () => clearTimeout(timeout);
            } else {
                // Finished typing command — proceed asynchronously
                setTimeout(() => {
                    setIsTyping(false);
                    setLines(prev => [...prev, { type: 'cmd', text: command.cmd }]);
                    setCurrentText('');
                    // Show output after a small delay
                    setTimeout(() => {
                        setLines(prev => [...prev, { type: 'output', text: command.output }]);
                        setCurrentCommandIndex(prev => prev + 1);
                        setIsTyping(true);
                    }, 500);
                }, 300);
            }
        }
    }, [currentText, currentCommandIndex, isTyping]);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [lines, currentText]);

    return (
        <div className="w-full rounded-xl overflow-hidden bg-[#1e1e1e] border border-neutral-800 shadow-2xl font-mono text-sm h-[300px] flex flex-col">
            {/* Title Bar */}
            <div className="bg-[#2d2d2d] px-4 py-2 flex items-center justify-between border-b border-neutral-700">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500" />
                    <div className="w-3 h-3 rounded-full bg-green-500" />
                </div>
                <div className="text-neutral-400 text-xs flex items-center gap-2">
                    <TerminalIcon className="w-3 h-3" />
                    <span>aether-cli — 80x24</span>
                </div>
                <div className="flex items-center gap-2 text-neutral-500">
                    <Minus className="w-3 h-3" />
                    <Maximize2 className="w-3 h-3" />
                    <X className="w-3 h-3" />
                </div>
            </div>

            {/* Content */}
            <div ref={scrollRef} className="p-4 text-neutral-300 flex-1 overflow-y-auto space-y-2 scrollbar-thin scrollbar-thumb-neutral-700">
                <div className="text-neutral-500 mb-4">Welcome to Aether AI CLI v2.0.0</div>

                {lines.map((line, idx) => (
                    <div key={idx} className={`${line.type === 'cmd' ? 'text-white' : 'text-neutral-400 whitespace-pre-wrap'}`}>
                        {line.type === 'cmd' && <span className="text-green-500 mr-2">➜</span>}
                        {line.type === 'cmd' && <span className="text-cyan-500 mr-2">~</span>}
                        {line.text}
                    </div>
                ))}

                {isTyping && (
                    <div className="text-white">
                        <span className="text-green-500 mr-2">➜</span>
                        <span className="text-cyan-500 mr-2">~</span>
                        {currentText}
                        <span className="animate-pulse inline-block w-2 h-4 bg-white ml-1 align-middle"></span>
                    </div>
                )}

                {!isTyping && currentCommandIndex < COMMANDS.length && (
                    <div className="text-white">
                        <span className="text-green-500 mr-2">➜</span>
                        <span className="text-cyan-500 mr-2">~</span>
                        <span className="animate-pulse inline-block w-2 h-4 bg-white ml-1 align-middle"></span>
                    </div>
                )}
            </div>
        </div>
    );
}
