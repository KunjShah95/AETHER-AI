import React from 'react';
import { Github, Twitter, Linkedin, Terminal, Heart } from 'lucide-react';

export function Footer() {
    return (
        <footer className="w-full pt-20 pb-10 bg-black border-t border-white/10 relative z-20">
            <div className="max-w-7xl mx-auto px-6">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">

                    {/* Brand Column */}
                    <div className="col-span-1 md:col-span-2 space-y-4">
                        <div className="flex items-center gap-3">
                            <div className="h-10 w-10 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-xl flex items-center justify-center text-white shadow-lg">
                                <Terminal size={20} strokeWidth={2.5} />
                            </div>
                            <span className="text-white font-heading font-bold text-2xl tracking-wide">
                                NEXUS AI
                            </span>
                        </div>
                        <p className="text-neutral-400 text-sm max-w-sm leading-relaxed">
                            The intelligent terminal assistant for modern engineering teams.
                            Seamlessly integrate local LLMs, streamline code reviews, and automate workflows—all from your CLI.
                        </p>
                    </div>

                    {/* Links Column 1 */}
                    <div className="space-y-4">
                        <h4 className="text-white font-semibold tracking-wide">Product</h4>
                        <ul className="space-y-2 text-sm text-neutral-400">
                            <li><a href="#features" className="hover:text-indigo-400 transition-colors">Features</a></li>
                            <li><a href="#download" className="hover:text-indigo-400 transition-colors">Download</a></li>
                            <li><a href="/docs" className="hover:text-indigo-400 transition-colors">Documentation</a></li>
                            <li><a href="#" className="hover:text-indigo-400 transition-colors">Changelog</a></li>
                        </ul>
                    </div>

                    {/* Links Column 2 */}
                    <div className="space-y-4">
                        <h4 className="text-white font-semibold tracking-wide">Community</h4>
                        <ul className="space-y-2 text-sm text-neutral-400">
                            <li><a href="https://github.com/KunjShah95/NEXUS-AI.io" className="hover:text-indigo-400 transition-colors">GitHub</a></li>
                            <li><a href="https://github.com/KunjShah95/NEXUS-AI.io/issues" className="hover:text-indigo-400 transition-colors">Issues</a></li>
                            <li><a href="#" className="hover:text-indigo-400 transition-colors">Discord (Soon)</a></li>
                        </ul>
                    </div>
                </div>

                <div className="border-t border-white/10 pt-8 flex flex-col md:flex-row justify-between items-center gap-6">
                    <p className="text-neutral-500 text-sm">
                        &copy; 2025 NEXUS AI. Distributed under MIT License.
                    </p>

                    <div className="flex items-center gap-6">
                        <a href="https://twitter.com/INDIA_KUNJ" target="_blank" rel="noopener noreferrer" className="text-neutral-400 hover:text-white transition-colors p-2 hover:bg-white/5 rounded-full">
                            <Twitter className="w-5 h-5" />
                        </a>
                        <a href="https://www.linkedin.com/in/kunjshah05/" target="_blank" rel="noopener noreferrer" className="text-neutral-400 hover:text-white transition-colors p-2 hover:bg-white/5 rounded-full">
                            <Linkedin className="w-5 h-5" />
                        </a>
                        <a href="https://github.com/KunjShah95" target="_blank" rel="noopener noreferrer" className="text-neutral-400 hover:text-white transition-colors p-2 hover:bg-white/5 rounded-full">
                            <Github className="w-5 h-5" />
                        </a>
                    </div>
                </div>

                <div className="mt-8 flex items-center justify-center gap-2 text-xs text-neutral-600">
                    <span>Made with</span>
                    <Heart className="w-3 h-3 text-red-500 fill-red-500 animate-pulse" />
                    <span>by Kunj Shah</span>
                </div>
            </div>
        </footer>
    );
}
