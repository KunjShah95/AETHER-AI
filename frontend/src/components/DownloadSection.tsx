import React, { useState } from 'react';
import { Download } from 'lucide-react';

export function DownloadSection() {
    const [downloadCount, setDownloadCount] = useState<number>(() => {
        const stored = typeof window !== 'undefined' ? localStorage.getItem('nexusDownloadCount') : null;
        return stored ? parseInt(stored) : 0;
    });

    const handleDownload = () => {
        const newCount = downloadCount + 1;
        setDownloadCount(newCount);
        localStorage.setItem('nexusDownloadCount', newCount.toString());
    };

    return (
        <section id="download" className="py-24 bg-black relative z-20 border-t border-white/[0.1]">
            <div className="max-w-4xl mx-auto px-6">
                <div className="text-center mb-12">
                    <h2 className="text-3xl md:text-5xl font-bold mb-4 text-white">Download AetherAI</h2>
                    <p className="text-lg text-neutral-400 max-w-2xl mx-auto">Get started with AetherAI on your platform</p>
                    <p className="text-sm text-terminal-cyan mt-2 font-mono">Total Downloads: {downloadCount.toLocaleString()}</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {['Windows', 'Linux', 'macOS'].map((os) => (
                        <div key={os} className="bg-neutral-900/50 backdrop-blur border border-white/[0.1] rounded-2xl p-8 text-center hover:border-terminal-cyan/50 transition-all group relative overflow-hidden">
                            <div className="absolute inset-0 bg-terminal-cyan/5 opacity-0 group-hover:opacity-100 transition-opacity" />

                            <div className="text-4xl mb-4 font-bold text-white relative z-10">
                                {os === 'Windows' && '🪟'}
                                {os === 'Linux' && '🐧'}
                                {os === 'macOS' && '🍎'}
                            </div>
                            <h3 className="text-xl font-semibold mb-2 text-white relative z-10">{os} Installer</h3>
                            <p className="text-neutral-400 mb-6 text-sm relative z-10">Automated setup script for {os}</p>

                            <div className="relative z-10">
                                <a
                                    href={`/dist/install_${os.toLowerCase()}.zip`}
                                    download
                                    onClick={handleDownload}
                                    className="inline-flex items-center justify-center gap-2 w-full px-6 py-3 rounded-lg bg-red-600 text-white font-bold hover:bg-red-700 transition-colors"
                                >
                                    <Download className="w-4 h-4" />
                                    <span>Download</span>
                                </a>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
