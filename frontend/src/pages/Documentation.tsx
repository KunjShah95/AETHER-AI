import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    BookOpen,
    Terminal,
    Zap,
    Shield,
    Palette,
    CheckSquare,
    Code,
    Rocket,
    Menu,
    X,
    Home
} from 'lucide-react';
import DocContent from '../components/docs/DocContent';
import Sidebar from '../components/docs/Sidebar';
import SearchBar from '../components/docs/SearchBar';

const Documentation = () => {
    const [activeSection, setActiveSection] = useState('home');
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');

    const sections = [
        { id: 'home', title: 'Home', icon: Home },
        { id: 'getting-started', title: 'Getting Started', icon: Rocket },
        { id: 'commands', title: 'Commands', icon: Terminal },
        { id: 'models', title: 'AI Models', icon: Zap },
        { id: 'advanced', title: 'Advanced Features', icon: Code },
        { id: 'tasks', title: 'Task Management', icon: CheckSquare },
        { id: 'themes', title: 'Themes', icon: Palette },
        { id: 'security', title: 'Security', icon: Shield },
        { id: 'developer', title: 'Developer Guide', icon: BookOpen },
    ];

    return (
        <div className="min-h-screen bg-black text-white">
            {/* Animated Background */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-0 left-0 w-full h-full bg-grid-white/[0.02]" />
                <div className="absolute -top-1/2 -left-1/2 w-full h-full bg-indigo-500/10 blur-3xl" />
            </div>

            {/* Header */}
            <motion.header
                initial={{ y: -100, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                className="sticky top-0 z-50 backdrop-blur-xl bg-slate-900/50 border-b border-purple-500/20"
            >
                <div className="max-w-screen-2xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <button
                                onClick={() => setSidebarOpen(!sidebarOpen)}
                                className="lg:hidden p-2 hover:bg-purple-500/20 rounded-lg transition-colors"
                            >
                                {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
                            </button>
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-blue-500 rounded-lg flex items-center justify-center">
                                    <BookOpen className="w-6 h-6" />
                                </div>
                                <div>
                                    <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                                        NEXUS AI Documentation
                                    </h1>
                                    <p className="text-xs text-slate-400">Production-ready AI Terminal Assistant</p>
                                </div>
                            </div>
                        </div>

                        <SearchBar
                            searchQuery={searchQuery}
                            setSearchQuery={setSearchQuery}
                            onNavigate={setActiveSection}
                        />
                    </div>
                </div>
            </motion.header>

            {/* Main Content */}
            <div className="max-w-screen-2xl mx-auto px-6 py-8">
                <div className="flex gap-8">
                    {/* Sidebar */}
                    <AnimatePresence>
                        {sidebarOpen && (
                            <Sidebar
                                sections={sections}
                                activeSection={activeSection}
                                setActiveSection={setActiveSection}
                            />
                        )}
                    </AnimatePresence>

                    {/* Content Area */}
                    <motion.main
                        layout
                        className="flex-1 min-w-0"
                    >
                        <DocContent
                            activeSection={activeSection}
                        />
                    </motion.main>
                </div>
            </div>

            {/* Footer */}
            <footer className="relative mt-20 border-t border-purple-500/20 backdrop-blur-xl bg-slate-900/30">
                <div className="max-w-screen-2xl mx-auto px-6 py-8">
                    <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                        <p className="text-slate-400 text-sm">
                            © 2025 NEXUS AI. Open source and free forever.
                        </p>
                        <div className="flex gap-6">
                            <a href="https://github.com/KunjShah95/NEXUS-AI.io"
                                className="text-slate-400 hover:text-purple-400 transition-colors text-sm">
                                GitHub
                            </a>
                            <a href="/" className="text-slate-400 hover:text-purple-400 transition-colors text-sm">
                                Home
                            </a>
                            <a href="#" className="text-slate-400 hover:text-purple-400 transition-colors text-sm">
                                License
                            </a>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default Documentation;
