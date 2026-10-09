#![no_main]

sp1_zkvm::entrypoint!(main);

fn main() {
    sp1_genguard_guest::guest_main();
}
