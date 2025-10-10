export default {
    root: 'src/',
    publicDir: '../static/',
    base: '/NASA-Space-App-Challenge-2025/',
    assetsInclude: [
        '**/*.glb',
        '**/*.gltf'
    ],
    server:
    {
        host: true, // Open to local network and display URL
        open: !('SANDBOX_URL' in process.env || 'CODESANDBOX_HOST' in process.env) // Open if it's not a CodeSandbox
    },
    build:
    {
        outDir: '../dist', // Output in the dist/ folder
        emptyOutDir: true, // Empty the folder first
        sourcemap: true, // Add sourcemap
        rollupOptions:
        {
            input:
            {
                main: 'src/index.html',
                empal: 'src/empal.html',
                quiz: 'src/quiz.html',
                rainfall: 'src/rainfall.html'
            }
        }
    },
}