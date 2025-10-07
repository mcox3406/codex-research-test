# Template structures

Place initial cyclic peptide structures here for Phase 1 generation scripts.
For example, create `cyclo_ala6_initial.pdb` using AMBER `tleap` or another
builder and save it into this directory before running the MD or Monte Carlo
pipelines.

Here’s a step-by-step guide to building the cyclic peptide using PyMOL commands.

You can copy and paste these commands directly into the PyMOL command line (it usually says `PyMOL>`).

1.  **Build the Linear Peptide:**
    First, we'll build a linear chain of six alanines. The `fab` command (fragment builder) is perfect for this. We'll name the object `ala6`.

    ```python
    fab AAAAAA, ala6
    ```

2.  **Display as Sticks:**
    This makes it easier to see the atoms you need to connect.

    ```python
    show sticks, ala6
    ```

3.  **Form the Cyclic Bond:** ⛓️
    This is the most important step. We will create a new covalent bond between the **nitrogen atom (`name N`) of the first residue (`resi 1`)** and the **carbonyl carbon atom (`name C`) of the sixth residue (`resi 6`)**.

    ```python
    bond (ala6 and resi 1 and name N), (ala6 and resi 6 and name C)
    ```

    When you run this command, you will see a long, stretched bond appear across the molecule, closing the loop.

4.  **Clean Up the Geometry:**
    The structure now has a very strained bond. The `clean` command will perform a quick energy minimization to relax the structure into a much more reasonable-looking ring.

    ```python
    clean ala6
    ```

    The molecule should now look like a proper 3D ring instead of a stretched-out chain.

5.  **Save the PDB File:**
    Finally, save your new cyclic structure to a PDB file.

    ```python
    save cyclo_ala6_initial.pdb, ala6
    ```

This will save the `cyclo_ala6_initial.pdb` file in the directory from which you launched PyMOL. You can then move it to your `data/templates/` folder. This method gives you visual confirmation at each step that you are building the correct molecule.