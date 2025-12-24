import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X, Github, Terminal } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/Button";

export const Navbar = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20);
        };
        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    const navItems = [
        { name: "Features", link: "/#features" },
        { name: "How it Works", link: "/#how-it-works" },
        { name: "Pricing", link: "/#pricing" },
        { name: "Docs", link: "/docs" },
    ];

    return (
        <nav
            className={cn(
                "fixed top-0 inset-x-0 z-50 transition-all duration-300 border-b",
                scrolled
                    ? "bg-black/80 backdrop-blur-md border-white/10"
                    : "bg-transparent border-transparent"
            )}
        >
            <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                {/* Logo */}
                <a href="/" className="flex items-center gap-2 group">
                    <div className="h-8 w-8 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-lg flex items-center justify-center text-white shadow-lg shadow-indigo-500/20 group-hover:scale-105 transition-transform">
                        <Terminal size={18} strokeWidth={3} />
                    </div>
                    <span className="text-white font-bold text-lg tracking-tight ml-1">
                        NEXUS
                        <span className="text-indigo-400">AI</span>
                    </span>
                </a>

                {/* Desktop Menu */}
                <div className="hidden md:flex items-center gap-8">
                    {navItems.map((item, idx) => (
                        <a
                            key={idx}
                            href={item.link}
                            className="text-sm font-medium text-neutral-400 hover:text-white transition-colors"
                        >
                            {item.name}
                        </a>
                    ))}
                </div>

                {/* Actions */}
                <div className="hidden md:flex items-center gap-4">
                    <a
                        href="https://github.com/KunjShah95/NEXUS-AI.io"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-neutral-400 hover:text-white transition-colors"
                    >
                        <Github size={20} />
                    </a>
                    <Button
                        size="sm"
                        className="bg-white text-black hover:bg-neutral-200 font-semibold"
                        onClick={() => document.getElementById('download')?.scrollIntoView({ behavior: 'smooth' })}
                    >
                        Get Started
                    </Button>
                </div>

                {/* Mobile Toggle */}
                <button
                    onClick={() => setIsOpen(!isOpen)}
                    className="md:hidden p-2 text-neutral-400 hover:text-white"
                >
                    {isOpen ? <X size={24} /> : <Menu size={24} />}
                </button>
            </div>

            {/* Mobile Menu */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="md:hidden bg-black/95 border-b border-white/10 overflow-hidden"
                    >
                        <div className="px-6 py-4 flex flex-col gap-4">
                            {navItems.map((item, idx) => (
                                <a
                                    key={idx}
                                    href={item.link}
                                    onClick={() => setIsOpen(false)}
                                    className="text-neutral-300 hover:text-white text-base font-medium py-2"
                                >
                                    {item.name}
                                </a>
                            ))}
                            <div className="h-px bg-white/10 my-2" />
                            <Button
                                className="w-full bg-white text-black hover:bg-neutral-200"
                                onClick={() => setIsOpen(false)}
                            >
                                Get Started
                            </Button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </nav>
    );
};
