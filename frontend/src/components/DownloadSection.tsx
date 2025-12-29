import React, { useState } from 'react';
import { Download, Monitor, Command, Terminal } from 'lucide-react';
import { Button } from "@/components/ui/Button";

export function DownloadSection() {
    const [downloadCount, setDownloadCount] = useState<number>(() => {
        const stored = typeof window !== 'undefined' ? localStorage.getItem('aetherDownloadCount') : null;
        return stored ? parseInt(stored) : 1240; // Fake starting number for social proof
    });

    const handleDownload = () => {
        const newCount = downloadCount + 1;
        setDownloadCount(newCount);
        localStorage.setItem('aetherDownloadCount', newCount.toString());
    };

    return (
        <section id="download" className="py-24 bg-black relative z-20 border-t border-white/[0.05]">
            <div className="max-w-6xl mx-auto px-6">
                <div className="text-center mb-16">
                    <h2 className="text-4xl md:text-5xl font-heading font-bold mb-6 text-white tracking-tight">
                        Install Anywhere
                    </h2>
                    <p className="text-lg text-neutral-400 max-w-2xl mx-auto">
                        Native performance on every major operating system.
                    </p>
                    <div className="mt-4 px-4 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-sm font-mono inline-flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></span>
                        {downloadCount.toLocaleString()} downloads this week
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {[
                        { name: 'Windows', icon: Monitor, cmd: 'irm aether.ai/win | iex', file: 'install_windows.bat' },
                        { name: 'Linux', icon: Terminal, cmd: 'curl -sL aether.ai/lin | bash', file: 'install_linux.sh' },
                        { name: 'macOS', icon: Command, cmd: 'curl -sL aether.ai/mac | bash', file: 'install_mac.sh' }
                    ].map((os) => (
                        <div key={os.name} className="bg-neutral-900/40 backdrop-blur border border-white/10 rounded-3xl p-8 text-center hover:border-indigo-500/50 transition-all duration-300 group relative overflow-hidden flex flex-col">
                            <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                            <div className="mb-6 flex justify-center relative z-10">
                                <div className="h-16 w-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center group-hover:scale-110 transition-transform duration-300 shadow-xl">
                                    <os.icon className="w-8 h-8 text-white" />
                                </div>
                            </div>

                            <h3 className="text-xl font-heading font-bold mb-2 text-white relative z-10">{os.name}</h3>

                            <div className="bg-black/50 rounded-lg p-3 mb-6 border border-white/5 relative z-10">
                                <code className="text-xs text-neutral-400 font-mono break-all">{os.cmd}</code>
                            </div>

                            <div className="relative z-10 mt-auto">
                                <Button
                                    className="bg-indigo-600 text-white font-semibold w-full h-12 hover:bg-indigo-700 shadow-lg shadow-indigo-500/10 rounded-xl"
                                    onClick={handleDownload}
                                >
                                    <div className="flex items-center gap-2">
                                        <Download className="w-4 h-4" />
                                        <span>Download Script</span>
                                    </div>
                                </Button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
