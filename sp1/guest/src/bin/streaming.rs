#![no_main]

sp1_zkvm::entrypoint!(main);

fn main() {
    sp1_genguard_guest::streaming_guest_main();
}
