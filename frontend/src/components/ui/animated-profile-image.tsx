'use client'
import Image from "next/image";

interface AnimatedProfileImageProps {
    src: string;
    alt: string;
    width: number;
    height: number;
    className?: string;
}

export function AnimatedProfileImage({ src, alt, width, height, className }: AnimatedProfileImageProps) {
    return (
        <div className="relative">
            {/* Simple circular border */}
            <div className="absolute -inset-[3px] bg-zinc-200 dark:bg-zinc-800 rounded-full" />
            
            {/* Profile Image Container */}
            <div className={`relative rounded-full overflow-hidden border-4 border-white dark:border-zinc-950 bg-zinc-100 dark:bg-zinc-900 ${className || ''}`}>
                <Image
                    src={src}
                    alt={alt}
                    width={width}
                    height={height}
                    priority
                    className="w-full h-full object-cover"
                />
            </div>
        </div>
    );
}
