import React from "react";
import { BentoGrid, BentoGridItem } from "@/components/ui/BentoGrid";
import {
    Brain,
    Terminal as TerminalIcon,
    ListTodo,
    Palette,
    Code2,
    Plug,
    Shield,
} from "lucide-react";

export function Features() {
    return (
        <section id="features" className="py-24 bg-black relative z-20">
            <div className="max-w-7xl mx-auto px-6">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-5xl font-bold mb-4 text-white">
                        Core Features
                    </h2>
                    <p className="text-lg text-neutral-400 max-w-2xl mx-auto">
                        Built for developers who need speed, privacy, and control.
                    </p>
                </div>

                <BentoGrid className="max-w-6xl mx-auto">
                    {items.map((item, i) => (
                        <BentoGridItem
                            key={i}
                            title={item.title}
                            description={item.description}
                            header={item.header}
                            icon={item.icon}
                            className={i === 3 || i === 6 ? "md:col-span-2" : ""}
                        />
                    ))}
                </BentoGrid>
            </div>
        </section>
    );
}

const FeatureHeader = ({ icon: Icon, color }: { icon: React.ComponentType<{ className?: string }>; color: string }) => (
    <div className={`flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br ${color} border border-white/[0.1] items-center justify-center`}>
        <Icon className="h-12 w-12 text-white/80" />
    </div>
);

const items = [
    {
        title: "Multi-Model Support",
        description: "Seamlessly switch between 8+ AI models: Gemini, Groq, DeepSeek, and more.",
        header: <FeatureHeader icon={Brain} color="from-cyan-600/40 to-blue-600/40" />,
        icon: <Brain className="h-4 w-4 text-neutral-300" />,
    },
    {
        title: "Terminal Interface",
        description: "Powerful CLI with syntax highlighting and auto-completion.",
        header: <FeatureHeader icon={TerminalIcon} color="from-terminal-cyan/40 to-cyan-600/40" />,
        icon: <TerminalIcon className="h-4 w-4 text-neutral-300" />,
    },
    {
        title: "Task Management",
        description: "Complete system with categories, priorities, and tracking.",
        header: <FeatureHeader icon={ListTodo} color="from-green-600/40 to-emerald-600/40" />,
        icon: <ListTodo className="h-4 w-4 text-neutral-300" />,
    },
    {
        title: "Code Review Assistant",
        description:
            "AI-powered code analysis supporting 15+ languages with security, performance, and quality assessments.",
        header: <FeatureHeader icon={Code2} color="from-orange-600/40 to-red-600/40" />,
        icon: <Code2 className="h-4 w-4 text-neutral-300" />,
    },
    {
        title: "Customizable Themes",
        description: "6 built-in themes plus custom theme creation.",
        header: <FeatureHeader icon={Palette} color="from-pink-600/40 to-rose-600/40" />,
        icon: <Palette className="h-4 w-4 text-neutral-300" />,
    },
    {
        title: "Integration Hub",
        description: "Connect with GitHub, Slack, Discord, Trello, Jira.",
        header: <FeatureHeader icon={Plug} color="from-yellow-600/40 to-amber-600/40" />,
        icon: <Plug className="h-4 w-4 text-neutral-300" />,
    },
    {
        title: "Secure & Private",
        description: "Your data stays local with enterprise-grade security. No data leaves your machine without your permission.",
        header: <FeatureHeader icon={Shield} color="from-slate-600/40 to-zinc-600/40" />,
        icon: <Shield className="h-4 w-4 text-neutral-300" />,
    },
];
