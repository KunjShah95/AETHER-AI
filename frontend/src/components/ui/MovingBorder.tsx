import React from "react";
import {
    motion,
    useAnimationFrame,
    useMotionValue,
    useTransform,
} from "framer-motion";
import { useRef } from "react";
import { cn } from "@/lib/utils";
import "./MovingBorder.css";

export function Button({
    borderRadius = "1.75rem",
    children,
    as: Component = "button",
    containerClassName,
    borderClassName,
    duration,
    className,
    ...otherProps
}: {
    borderRadius?: string;
    children: React.ReactNode;
    as?: React.ElementType;
    containerClassName?: string;
    borderClassName?: string;
    duration?: number;
    className?: string;
    [key: string]: unknown;
}) {
    return (
        <Component
            className={cn(
                "bg-transparent relative text-xl  h-16 w-40 p-[1px] overflow-hidden moving-border-container",
                containerClassName
            )}
            style={{
                "--border-radius": borderRadius,
            }}
            {...otherProps}
        >
            <div
                className="absolute inset-0 moving-border-inner"
            >
                <MovingBorder duration={duration} rx="30%" ry="30%">
                    <div
                        className={cn(
                            "h-20 w-20 opacity-[0.8] bg-[radial-gradient(var(--sky-500)_40%,transparent_60%)]",
                            borderClassName
                        )}
                    />
                </MovingBorder>
            </div>

            <div
                className={cn(
                    "relative bg-slate-900/[0.8] border border-slate-800 backdrop-blur-xl text-white flex items-center justify-center w-full h-full text-sm antialiased moving-border-content",
                    className
                )}
            >
                {children}
            </div>
        </Component>
    );
}

export const MovingBorder = ({
    children,
    duration = 2000,
    rx,
    ry,
    ...otherProps
}: {
    children: React.ReactNode;
    duration?: number;
    rx?: string;
    ry?: string;
    [key: string]: unknown;
}) => {
    const pathRef = useRef<SVGRectElement>(null);
    const progress = useMotionValue<number>(0);

    useAnimationFrame((time) => {
        const length = pathRef.current?.getTotalLength();
        if (length) {
            const pxPerMillisecond = length / duration;
            progress.set((time * pxPerMillisecond) % length);
        }
    });

    const x = useTransform(
        progress,
        (val) => pathRef.current?.getPointAtLength(val).x
    );
    const y = useTransform(
        progress,
        (val) => pathRef.current?.getPointAtLength(val).y
    );

    return (
        <>
            <svg
                xmlns="http://www.w3.org/2000/svg"
                preserveAspectRatio="none"
                className="absolute h-full w-full"
                width="100%"
                height="100%"
                {...otherProps}
            >
                <rect
                    fill="none"
                    width="100%"
                    height="100%"
                    rx={rx}
                    ry={ry}
                    ref={pathRef}
                />
            </svg>
            <motion.div
                className="moving-border-animation"
                style={
                    {
                        "--x": `${x.get()}px`,
                        "--y": `${y.get()}px`,
                    } as React.CSSProperties
                }
            >
                {children}
            </motion.div>
        </>
    );
};
