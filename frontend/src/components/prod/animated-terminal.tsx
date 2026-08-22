'use client';
import { motion } from "motion/react";
import { cn } from "@/lib/utils";

interface AnimatedTerminalProps {
    children: React.ReactNode;
    className?: string;
    title?: string;
}

export function AnimatedTerminal({ children, className, title = "root@ahmed-bargady:~" }: AnimatedTerminalProps) {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className={cn(
                "relative w-full max-w-5xl mx-auto overflow-hidden rounded-xl border border-slate-800 bg-slate-950/90 shadow-2xl backdrop-blur-md",
                className
            )}
        >
            {/* Terminal Header */}
            <div className="flex items-center justify-between border-b border-slate-800 bg-slate-900/50 px-4 py-3">
                <div className="flex items-center gap-2">
                    <div className="flex gap-2">
                        <div className="h-3 w-3 rounded-full bg-red-500/80" />
                        <div className="h-3 w-3 rounded-full bg-amber-500/80" />
                        <div className="h-3 w-3 rounded-full bg-emerald-500/80" />
                    </div>
                    <div className="ml-4 font-mono text-xs text-slate-400 select-none">
                        {title}
                    </div>
                </div>
                <div className="font-mono text-xs text-slate-600">
                    bash -- 80x24
                </div>
            </div>

            {/* Scanline Effect */}
            <div className="pointer-events-none absolute inset-0 z-50 overflow-hidden opacity-10 mix-blend-overlay">
                 <div className="absolute inset-0 bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))]" style={{ backgroundSize: '100% 4px, 6px 100%' }}></div>
            </div>

            {/* Content */}
            <div className="relative z-10 font-mono flex-grow flex flex-col overflow-hidden">
                {children}
            </div>
        </motion.div>
    );
}
