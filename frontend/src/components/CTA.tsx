import React from "react";
import { CardStack } from "@/components/ui/CardStack";
import { cn } from "@/lib/utils";

export function CTA() {
    return (
        <div className="h-[40rem] flex items-center justify-center w-full bg-black relative z-20">
            <div className="flex flex-col md:flex-row items-center justify-center gap-20 max-w-7xl mx-auto px-6">

                <div className="max-w-xl">
                    <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">
                        Open Source.<br />
                        <span className="text-terminal-cyan">Always Free.</span>
                    </h2>
                    <p className="text-lg text-neutral-400 mb-8">
                        Join developers who keep their code local and their workflow fast.
                    </p>
                    <div className="flex gap-4">
                        <button className="px-8 py-3 rounded-full bg-terminal-cyan text-black font-bold hover:bg-terminal-cyan/90 transition-colors">
                            Join Discord
                        </button>
                        <button className="px-8 py-3 rounded-full bg-transparent border border-neutral-700 text-white font-bold hover:bg-neutral-800 transition-colors">
                            Read Docs
                        </button>
                    </div>
                </div>

                <CardStack items={CARDS} />
            </div>
        </div>
    );
}

// Small utility to highlight the content of specific section of a testimonial content
export const Highlight = ({
    children,
    className,
}: {
    children: React.ReactNode;
    className?: string;
}) => {
    return (
        <span
            className={cn(
                "font-bold bg-emerald-100 text-emerald-700 dark:bg-emerald-700/[0.2] dark:text-emerald-500 px-1 py-0.5",
                className
            )}
        >
            {children}
        </span>
    );
};

const CARDS = [
    {
        id: 0,
        name: "Alex Chen",
        designation: "Senior Engineer at TechCorp",
        content: (
            <p>
                AetherAI is a game changer. I can finally use LLMs on my proprietary code without worrying about <Highlight>data leaks</Highlight>. The local model support is blazing fast.
            </p>
        ),
    },
    {
        id: 1,
        name: "Sarah Jones",
        designation: "Freelance Developer",
        content: (
            <p>
                I used to pay $20/month for Copilot. Now I use <Highlight>AetherAI</Highlight> with Llama 3 locally and it's free, faster, and I own my data.
            </p>
        ),
    },
    {
        id: 2,
        name: "David Kim",
        designation: "Open Source Maintainer",
        content: (
            <p>
                The terminal integration is seamless. I love how I can just type <Highlight>aether chat</Highlight> and get answers right in my workflow.
            </p>
        ),
    },
];
