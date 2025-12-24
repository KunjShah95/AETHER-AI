import { motion } from 'framer-motion';
import type { LucideIcon } from 'lucide-react';

interface Section {
    id: string;
    title: string;
    icon: LucideIcon;
}

interface SidebarProps {
    sections: Section[];
    activeSection: string;
    setActiveSection: (id: string) => void;
}

const Sidebar = ({ sections, activeSection, setActiveSection }: SidebarProps) => {
    return (
        <motion.aside
            initial={{ x: -100, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -100, opacity: 0 }}
            className="w-64 lg:w-72 shrink-0"
        >
            <div className="sticky top-24">
                <div className="backdrop-blur-xl bg-slate-900/50 border border-blue-500/20 rounded-2xl p-6 shadow-2xl">
                    <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
                        Navigation
                    </h2>

                    <nav className="space-y-1">
                        {sections.map((section, index) => {
                            const Icon = section.icon;
                            const isActive = activeSection === section.id;

                            return (
                                <motion.button
                                    key={section.id}
                                    initial={{ x: -20, opacity: 0 }}
                                    animate={{ x: 0, opacity: 1 }}
                                    transition={{ delay: index * 0.05 }}
                                    onClick={() => setActiveSection(section.id)}
                                    className={`
                    w-full flex items-center gap-3 px-4 py-3 rounded-xl
                    transition-all duration-300 group relative overflow-hidden
                    ${isActive
                                            ? 'bg-gradient-to-r from-blue-500/20 to-indigo-500/20 text-white border border-blue-500/30'
                                            : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                                        }
                  `}
                                >
                                    {/* Active indicator */}
                                    {isActive && (
                                        <motion.div
                                            layoutId="activeSection"
                                            className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-blue-500 to-indigo-500 rounded-r"
                                        />
                                    )}

                                    {/* Icon */}
                                    <div className={`
                    w-8 h-8 rounded-lg flex items-center justify-center
                    transition-all duration-300
                    ${isActive
                                            ? 'bg-gradient-to-br from-blue-500 to-indigo-500 shadow-lg shadow-blue-500/50'
                                            : 'bg-slate-800/50 group-hover:bg-slate-700/50'
                                        }
                  `}>
                                        <Icon size={16} />
                                    </div>

                                    {/* Title */}
                                    <span className="font-medium text-sm flex-1 text-left">
                                        {section.title}
                                    </span>

                                    {/* Hover effect */}
                                    <div className="w-1 h-1 rounded-full bg-blue-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                                </motion.button>
                            );
                        })}
                    </nav>
                </div>

                {/* Quick Links */}
                <motion.div
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="mt-6 backdrop-blur-xl bg-gradient-to-br from-blue-500/10 to-indigo-500/10 border border-blue-500/20 rounded-2xl p-6"
                >
                    <h3 className="text-sm font-semibold text-slate-300 mb-3">Quick Links</h3>
                    <div className="space-y-2 text-sm">
                        <a href="https://github.com/KunjShah95/NEXUS-AI.io"
                            className="block text-slate-400 hover:text-blue-400 transition-colors">
                            → GitHub Repository
                        </a>
                        <a href="/"
                            className="block text-slate-400 hover:text-blue-400 transition-colors">
                            → Main Website
                        </a>
                        <a href="#"
                            className="block text-slate-400 hover:text-blue-400 transition-colors">
                            → Changelog
                        </a>
                    </div>
                </motion.div>
            </div>
        </motion.aside>
    );
};

export default Sidebar;
