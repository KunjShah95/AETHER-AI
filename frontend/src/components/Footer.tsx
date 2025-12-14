import React from 'react';
import { Github, Twitter, Linkedin } from 'lucide-react';

export function Footer() {
    return (
        <footer className="w-full py-12 bg-black border-t border-white/[0.1] relative z-20">
            <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">

                <div className="flex flex-col items-center md:items-start">
                    <div className="flex items-center gap-2 mb-2">
                        <div className="h-8 w-8 bg-terminal-cyan rounded-lg flex items-center justify-center">
                            <span className="text-black font-bold text-sm">A</span>
                        </div>
                        <span className="text-white font-bold text-xl tracking-wide">AETHER AI</span>
                    </div>
                    <p className="text-neutral-500 text-sm text-center md:text-left max-w-xs">
                        The next-generation AI terminal assistant. Open source, secure, and local.
                    </p>
                </div>

                <div className="flex gap-6">
                    <a href="https://twitter.com/INDIA_KUNJ" target="_blank" rel="noopener noreferrer" title="Follow us on Twitter" className="text-neutral-400 hover:text-white transition-colors">
                        <Twitter className="w-5 h-5" />
                    </a>
                    <a href="https://www.linkedin.com/in/kunjshah05/" target="_blank" rel="noopener noreferrer" title="Connect on LinkedIn" className="text-neutral-400 hover:text-white transition-colors">
                        <Linkedin className="w-5 h-5" />
                    </a>
                    <a href="https://github.com/KunjShah95" target="_blank" rel="noopener noreferrer" title="View our GitHub" className="text-neutral-400 hover:text-white transition-colors">
                        <Github className="w-5 h-5" />
                    </a>
                </div>

                <div className="text-neutral-500 text-sm">
                    &copy; 2025 AetherAI. All rights reserved.
                </div>
            </div>
        </footer>
    );
}
