// Application that needs function renaming

function getData() {
    return [1, 2, 3, 4, 5];
}

function main() {
    const data = getData();
    console.log("Data:", data);
}

main();

module.exports = { getData };
