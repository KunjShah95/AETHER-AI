import { useEffect } from "react";
import { motion, stagger, useAnimate } from "framer-motion";
import { cn } from "@/lib/utils";

export const TextGenerateEffect = ({
    words,
    className,
}: {
    words: string;
    className?: string;
}) => {
    const [scope, animate] = useAnimate();
    const wordsArray = words.split(" ");
    useEffect(() => {
        // Trigger animation after mount when DOM is available
        const id = setTimeout(() => {
            if (scope.current) {
                animate(
                    "span",
                    {
                        opacity: 1,
                    },
                    {
                        duration: 2,
                        delay: stagger(0.2),
                    }
                );
            }
        }, 0);
        return () => clearTimeout(id);
    // We intentionally only depend on `animate` here; `scope` is a ref and
    // including `scope` in the deps is unnecessary and noisy for this usage.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [animate]);

    const renderWords = () => {
        return (
            <motion.div ref={scope}>
                {wordsArray.map((word, idx) => {
                    return (
                        <motion.span
                            key={word + idx}
                            className="dark:text-white text-black opacity-0"
                        >
                            {word}{" "}
                        </motion.span>
                    );
                })}
            </motion.div>
        );
    };

    return (
        <div className={cn("font-bold", className)}>
            <div className="mt-4">
                <div className=" dark:text-white text-black text-2xl leading-snug tracking-wide">
                    {renderWords()}
                </div>
            </div>
        </div>
    );
};
