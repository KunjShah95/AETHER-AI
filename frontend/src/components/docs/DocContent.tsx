import { motion } from 'framer-motion';
import HomeDoc from './sections/HomeDoc.tsx';
import GettingStartedDoc from './sections/GettingStartedDoc.tsx';
import CommandsDoc from './sections/CommandsDoc.tsx';
import ModelsDoc from './sections/ModelsDoc.tsx';
import AdvancedDoc from './sections/AdvancedDoc.tsx';
import TasksDoc from './sections/TasksDoc.tsx';
import ThemesDoc from './sections/ThemesDoc.tsx';
import SecurityDoc from './sections/SecurityDoc.tsx';
import DeveloperDoc from './sections/DeveloperDoc.tsx';

interface DocContentProps {
    activeSection: string;
}

const DocContent = ({ activeSection }: DocContentProps) => {
    const renderSection = () => {
        switch (activeSection) {
            case 'home':
                return <HomeDoc />;
            case 'getting-started':
                return <GettingStartedDoc />;
            case 'commands':
                return <CommandsDoc />;
            case 'models':
                return <ModelsDoc />;
            case 'advanced':
                return <AdvancedDoc />;
            case 'tasks':
                return <TasksDoc />;
            case 'themes':
                return <ThemesDoc />;
            case 'security':
                return <SecurityDoc />;
            case 'developer':
                return <DeveloperDoc />;
            default:
                return <HomeDoc />;
        }
    };

    return (
        <motion.div
            key={activeSection}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
            className="backdrop-blur-xl bg-slate-900/50 border border-purple-500/20 rounded-2xl p-8 lg:p-12 shadow-2xl"
        >
            {renderSection()}
        </motion.div>
    );
};

export default DocContent;
