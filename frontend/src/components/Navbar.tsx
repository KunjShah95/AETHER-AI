import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { Menu, X, Github } from "lucide-react";

export const Navbar = () => {
    const [isOpen, setIsOpen] = useState(false);

    const navItems = [
        { name: "Home", link: "/" },
        { name: "Features", link: "/#features" },
        { name: "Download", link: "/#download" },
        { name: "Docs", link: "/docs" },
    ];

    return (
        <div className="fixed top-10 inset-x-0 max-w-2xl mx-auto z-50">
            <div className="relative flex items-center justify-between p-4 rounded-full border border-white/[0.2] bg-black/50 backdrop-blur-md shadow-[0px_2px_3px_-1px_rgba(0,0,0,0.1),0px_1px_0px_0px_rgba(25,28,33,0.02),0px_0px_0px_1px_rgba(25,28,33,0.08)]">

                {/* Logo */}
                <div className="flex items-center gap-2 px-4">
                    <div className="h-6 w-6 bg-terminal-cyan rounded-lg flex items-center justify-center">
                        <span className="text-black font-bold text-xs">A</span>
                    </div>
                    <span className="text-white font-bold text-sm tracking-wide">AETHER AI</span>
                </div>

                {/* Desktop Menu */}
                <div className="hidden md:flex items-center gap-4">
                    {navItems.map((item, idx) => (
                        <a
                            key={idx}
                            href={item.link}
                            className="text-neutral-300 hover:text-white text-sm font-medium transition-colors px-3 py-1 rounded-full hover:bg-white/[0.1]"
                        >
                            {item.name}
                        </a>
                    ))}
                </div>

                {/* CTA / GitHub */}
                <div className="hidden md:flex items-center px-4">
                    <a
                        href="https://github.com/KunjShah95/NEXUS-AI.io"
                        target="_blank"
                        className="flex items-center gap-2 text-xs font-medium text-white bg-white/[0.1] hover:bg-white/[0.2] border border-white/[0.2] px-4 py-2 rounded-full transition-all"
                    >
                        <Github className="w-4 h-4" />
                        <span>Star on GitHub</span>
                    </a>
                </div>

                {/* Mobile Toggle */}
                <div className="md:hidden px-4">
                    <button onClick={() => setIsOpen(!isOpen)} className="text-white">
                        {isOpen ? <X /> : <Menu />}
                    </button>
                </div>
            </div>

            {/* Mobile Menu */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -20, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -20, scale: 0.95 }}
                        className="absolute top-full left-0 right-0 mt-2 p-4 rounded-2xl bg-black/90 border border-white/[0.2] backdrop-blur-xl md:hidden"
                    >
                        <div className="flex flex-col gap-2">
                            {navItems.map((item, idx) => (
                                <a
                                    key={idx}
                                    href={item.link}
                                    onClick={() => setIsOpen(false)}
                                    className="text-neutral-300 hover:text-white text-sm font-medium p-3 rounded-xl hover:bg-white/[0.1] transition-colors"
                                >
                                    {item.name}
                                </a>
                            ))}
                            <div className="h-px bg-white/[0.1] my-2" />
                            <a
                                href="https://github.com/KunjShah95/NEXUS-AI.io"
                                target="_blank"
                                className="flex items-center justify-center gap-2 text-sm font-medium text-white bg-white/[0.1] p-3 rounded-xl"
                            >
                                <Github className="w-4 h-4" />
                                <span>Star on GitHub</span>
                            </a>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
