'use client'
import { motion } from "motion/react";
import { PlaceholdersAndVanishInput } from "@/components/ui/placeholders-and-vanish-input";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useStore } from "@/store/useStore";

export function AnimatedChatInput() {
    const router = useRouter();
    const { setInitialMessage } = useStore();
    const [inputValue, setInputValue] = useState("");

    const placeholders = [
        "Ask me anything about my work...",
        "What projects have you worked on?",
        "Tell me about your AI research",
        "What are your career goals?",
        "How can we collaborate?",
        "What technologies do you use?",
    ];

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setInputValue(e.target.value);
    };

    const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (inputValue.trim()) {
            setInitialMessage(inputValue.trim());
            router.push("/chat");
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ 
                duration: 0.6, 
                delay: 0.5,
                ease: [0.16, 1, 0.3, 1],
            }}
            className="w-full max-w-2xl mx-auto"
        >
            {/* Decorative Label */}
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.6 }}
                className="text-center mb-4"
            >
                <span className="text-sm text-zinc-500 dark:text-zinc-400 font-medium">
                    Chat with me to learn more
                </span>
            </motion.div>

            {/* Input Container - Minimalist Style */}
            <motion.div
                whileHover={{ scale: 1.01 }}
                transition={{ type: "spring", stiffness: 400, damping: 17 }}
                className="relative"
            >
                {/* Input Wrapper - Clean minimal style */}
                <div className="relative bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-full overflow-hidden shadow-sm hover:shadow-md transition-shadow duration-200">
                    <PlaceholdersAndVanishInput
                        placeholders={placeholders}
                        onChange={handleChange}
                        onSubmit={onSubmit}
                    />
                </div>
            </motion.div>

            {/* Hint Text */}
            <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1 }}
                className="text-center mt-3 text-xs text-zinc-400 dark:text-zinc-600"
            >
                Powered by AI • Ask anything about my experience
            </motion.p>
        </motion.div>
    );
}
