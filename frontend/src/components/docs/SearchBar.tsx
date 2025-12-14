import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Command, ArrowRight, X } from 'lucide-react';
import { documentationIndex } from '@/data/documentationIndex';
import { searchItems, getExcerpt } from '@/utils/searchUtils';

interface SearchBarProps {
    searchQuery: string;
    setSearchQuery: (query: string) => void;
    onNavigate?: (section: string) => void;
}

const SearchBar = ({ searchQuery, setSearchQuery, onNavigate }: SearchBarProps) => {
    const [isOpen, setIsOpen] = useState(false);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const searchRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // Perform search
    const searchResults = searchItems(documentationIndex, searchQuery).slice(0, 8);

    // Handle click outside to close
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Open dropdown when typing
    useEffect(() => {
        if (searchQuery.length > 0) {
            // Schedule open asynchronously to avoid sync setState in effect
            const id = setTimeout(() => {
                setIsOpen(true);
                setSelectedIndex(0);
            }, 0);
            return () => clearTimeout(id);
        } else {
            const id = setTimeout(() => setIsOpen(false), 0);
            return () => clearTimeout(id);
        }
    }, [searchQuery]);

    // Keyboard navigation
    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (!isOpen || searchResults.length === 0) return;

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setSelectedIndex(prev =>
                    prev < searchResults.length - 1 ? prev + 1 : prev
                );
                break;
            case 'ArrowUp':
                e.preventDefault();
                setSelectedIndex(prev => prev > 0 ? prev - 1 : 0);
                break;
            case 'Enter':
                e.preventDefault();
                if (searchResults[selectedIndex]) {
                    handleResultClick(searchResults[selectedIndex].item.section);
                }
                break;
            case 'Escape':
                e.preventDefault();
                setIsOpen(false);
                inputRef.current?.blur();
                break;
        }
    };

    // Handle result click
    const handleResultClick = (section: string) => {
        if (onNavigate) {
            onNavigate(section);
        }
        setSearchQuery('');
        setIsOpen(false);
        inputRef.current?.blur();
    };

    // Clear search
    const handleClear = () => {
        setSearchQuery('');
        setIsOpen(false);
        inputRef.current?.focus();
    };

    // Get category badge color
    const getCategoryColor = (category?: string) => {
        if (!category) return 'bg-slate-700/50 text-slate-300';

        const colorMap: Record<string, string> = {
            'System Commands': 'bg-purple-500/20 text-purple-300 border-purple-500/30',
            'Utility Commands': 'bg-blue-500/20 text-blue-300 border-blue-500/30',
            'Developer Commands': 'bg-green-500/20 text-green-300 border-green-500/30',
            'Git Advanced Commands': 'bg-orange-500/20 text-orange-300 border-orange-500/30',
            'Context-Aware AI': 'bg-purple-500/20 text-purple-300 border-purple-500/30',
            'Analytics & Monitoring': 'bg-blue-500/20 text-blue-300 border-blue-500/30',
            'Games & Learning': 'bg-green-500/20 text-green-300 border-green-500/30',
            'Creative Tools': 'bg-pink-500/20 text-pink-300 border-pink-500/30',
            'User Management': 'bg-orange-500/20 text-orange-300 border-orange-500/30',
            'Key Features': 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
        };

        return colorMap[category] || 'bg-slate-700/50 text-slate-300 border-slate-600/30';
    };

    return (
        <div className="relative hidden md:block" ref={searchRef}>
            <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4 pointer-events-none" />
                <input
                    ref={inputRef}
                    type="text"
                    placeholder="Search documentation..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onFocus={() => searchQuery && setIsOpen(true)}
                    className="
                        w-64 pl-10 pr-10 py-2 
                        bg-slate-800/50 backdrop-blur-xl 
                        border border-purple-500/20 
                        rounded-xl 
                        text-sm text-white placeholder-slate-400
                        focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50
                        focus:w-80
                        transition-all duration-300
                    "
                />
                {searchQuery && (
                    <button
                        type="button"
                        title="Clear search"
                        aria-label="Clear search"
                        onClick={handleClear}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-white transition-colors"
                    >
                        <X className="w-4 h-4" />
                    </button>
                )}
            </div>

            {/* Search results dropdown */}
            <AnimatePresence>
                {isOpen && searchQuery && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.2 }}
                        className="absolute top-full mt-2 w-[500px] bg-slate-900/98 backdrop-blur-xl border border-purple-500/30 rounded-xl shadow-2xl overflow-hidden z-50"
                    >
                        {searchResults.length > 0 ? (
                            <>
                                <div className="px-4 py-2 border-b border-purple-500/10 bg-slate-800/50">
                                    <p className="text-xs text-slate-400 flex items-center gap-2">
                                        <Command className="w-3 h-3" />
                                        Found {searchResults.length} result{searchResults.length !== 1 ? 's' : ''} for "{searchQuery}"
                                    </p>
                                </div>
                                <div className="max-h-[400px] overflow-y-auto">
                                    {searchResults.map((result, index) => (
                                        <motion.button
                                            key={result.item.id}
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: index * 0.03 }}
                                            onClick={() => handleResultClick(result.item.section)}
                                            onMouseEnter={() => setSelectedIndex(index)}
                                            className={`
                                                w-full text-left px-4 py-3 border-b border-purple-500/10 
                                                transition-all duration-200
                                                ${selectedIndex === index
                                                    ? 'bg-purple-500/20 border-l-2 border-l-purple-500'
                                                    : 'hover:bg-slate-800/50 border-l-2 border-l-transparent'
                                                }
                                            `}
                                        >
                                            <div className="flex items-start justify-between gap-3">
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex items-center gap-2 mb-1">
                                                        {result.item.category && (
                                                            <span className={`text-xs px-2 py-0.5 rounded border ${getCategoryColor(result.item.category)}`}>
                                                                {result.item.category}
                                                            </span>
                                                        )}
                                                        <span className="text-xs px-2 py-0.5 rounded bg-slate-700/50 text-slate-300">
                                                            {result.item.sectionTitle}
                                                        </span>
                                                    </div>
                                                    <h4 className="text-sm font-semibold text-white mb-1 flex items-center gap-2">
                                                        {result.item.title}
                                                        {result.item.title.startsWith('/') && (
                                                            <span className="text-xs text-purple-400 font-mono">command</span>
                                                        )}
                                                    </h4>
                                                    <p className="text-xs text-slate-400 line-clamp-2">
                                                        {getExcerpt(result.item.description, 120)}
                                                    </p>
                                                    <div className="flex items-center gap-2 mt-1">
                                                        <span className="text-xs text-purple-400/70">
                                                            Score: {result.score}
                                                        </span>
                                                        {result.matchedFields.length > 0 && (
                                                            <span className="text-xs text-slate-500">
                                                                • Matched: {result.matchedFields.slice(0, 2).join(', ')}
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                                <ArrowRight className="w-4 h-4 text-purple-400 shrink-0 mt-1" />
                                            </div>
                                        </motion.button>
                                    ))}
                                </div>
                                <div className="px-4 py-2 border-t border-purple-500/10 bg-slate-800/30">
                                    <p className="text-xs text-slate-500 flex items-center gap-2">
                                        <kbd className="px-1.5 py-0.5 bg-slate-700/50 rounded text-xs">↑↓</kbd> Navigate
                                        <kbd className="px-1.5 py-0.5 bg-slate-700/50 rounded text-xs">Enter</kbd> Select
                                        <kbd className="px-1.5 py-0.5 bg-slate-700/50 rounded text-xs">Esc</kbd> Close
                                    </p>
                                </div>
                            </>
                        ) : (
                            <div className="p-6 text-center">
                                <Search className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                                <p className="text-sm text-slate-400 mb-1">No results found</p>
                                <p className="text-xs text-slate-500">
                                    Try searching for commands, features, or topics
                                </p>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default SearchBar;
